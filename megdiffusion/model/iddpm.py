"""
Paper: `Improved Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2102.09672>`_ (a.k.a IDDPM)

The official PyTorch implementation: https://github.com/openai/improved-diffusion

Port from https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py

The model structure implemented here is as consistent as possible with the official implementation.
The differences between the code implementation and the paper description will be pointed out.

.. _dims-argument:

Directly use ``Conv2d()``,``AvgPool2d()`` instead of ``conv_nd(dims=2)``, ``avg_pool_nd(dims=2)`` and so on.
We only consider using 2-d feature data when reproduce the experiment in the original paper.

.. _checkpoint:

In official Pytorch implementation, use ``checkpoint`` for reduced memory, not apllied here:

https://github.com/openai/improved-diffusion/blob/mastetr/improved_diffusion/nn.py#L124

.. _fp16_type_convert:

Datatype convert between fp32 and fp16 are used for mixed precision training, not applied here.
"""

from abc import abstractmethod

import math

import megengine.functional as F
import megengine.module as M
import megengine.module.init as init

class SiLU(M.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
            These may be fractional.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    
    Return:
        an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = F.exp(
        -math.log(max_period) * F.arange(0, half, dtype="float32") / half
    )
    args = timesteps[:, None] * freqs[None]
    embedding = F.concat([F.cos(args), F.sin(args)], axis=-1)
    if dim % 2:
        embedding = F.concat([embedding, F.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

class TimestepBlock(M.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""

    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""
        

class TimestepEmbedSequential(M.Sequential, TimestepBlock):
    """A sequential module that passes timestep embeddings to the children that
    support it as an extra input."""

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class UpSample(M.Module):
    """An upsampling layer with an optional convolution.

    Args:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = M.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.nn.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(M.Module):
    """A downsampling layer with an optional convolution.

    Args:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = M.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = M.AvgPool2d(stride=2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """A residual block that can optionally change the number of channels.

    Args:
        channels: the number of input channels.
        emb_channels: the number of timestep embedding channels.
        dropout: the rate of dropout.
        out_channels: if specified, the number of out channels.
        use_conv: if True and out_channels is specified, use a spatial
            convolution instead of a smaller 1x1 convolution to change the
            channels in the skip connection.
    """

    def __init__(
        self, 
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = M.Sequential(
            M.GroupNorm(32, channels),
            SiLU(),
            M.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = M.Sequential(
            SiLU(),
            M.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels
            ),
        )
        self.out_layers = M.Sequential(
            M.GroupNorm(32, self.out_channels),
            SiLU(),
            M.Dropout(dropout),
            M.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        )

        # zero module
        for p in self.out_layers[-1].parameters():
            init.zeros_(p)

        if self.out_channels == channels:
            self.skip_connection = M.Identity()
        elif use_conv:
            self.skip_connection = M.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = M.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = F.split(emb_out, 2, axis=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(M.Module):
    """An attention block that allows spatial positions to attend to each other.

    Originally ported from here:
    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = M.GroupNorm(32, channels)
        self.qkv = M.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = M.Conv1d(channels, channels, 1)

        # zero module
        for p in self.proj_out.parameters():
            init.zeros_(p)

    def forward(self, x):
        b, c, *spatial = x.shape
        # MegEngine v1.9.1 ``GroupNorm`` only support NCHW input right now
        # So we do reshape after norm operation
        qkv = self.qkv(self.norm(x).reshape(b, c, -1))
        x = x.reshape(b, c, -1)
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)



class QKVAttention(M.Module):
    """A module which performs QKV attention."""

    def forward(self, qkv):
        """Apply QKV attention.

        Args:
            qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        
        Return:
            an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = F.split(qkv, 3, axis=1)   # Note: different with torch.split
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = (q.transpose(0, 2, 1) * scale) @ (k * scale)  # einsum("bct,bcs->bts", q * scale, k * scale)
        weight = F.nn.softmax(weight, axis=-1)
        return (v @ weight.transpose(0, 2, 1))  # einsum("bts,bcs->bct", weight, v)


class UNetModel(M.Module):
    """The full UNet model with attention and timestep embedding.
    
    Args:
        in_channels: channels in the input Tensor.
        model_channels: base channel count for the model.
        out_channels: channels in the output Tensor.
        num_res_blocks: number of residual blocks per downsample.
        attention_level: a collection of levels at which attention will take place.
        dropout: the dropout probability.
        channel_mult: channel multiplier for each level of the UNet.
        conv_resample: if True, use learned convolutions for upsampling and downsampling.
        num_classes: if specified (as an int), then this model will be
            class-conditional with `num_classes` classes.
        num_heads: the number of attention heads in each attention layer.
    
    Note:

        In original paper the argument ``attention_level`` is still named ``attension_resolution``
        like what DDPM code does. But there is ambiguity between the name and the actual purpose.
        So we rename it to ``attention_level`` making the description more accurate.
    """
    
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_level,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_classes=None,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_level = attention_level
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = M.Sequential(
            M.Linear(model_channels, time_embed_dim),
            SiLU(),
            M.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = M.Embedding(num_classes, time_embed_dim)

        self.input_blocks = [
            TimestepEmbedSequential(
                M.Conv2d(in_channels, model_channels, 3, padding=1)
            )
        ]

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels= mult * model_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_level:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm
            ),
        )

        self.output_blocks = []
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks+1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = model_channels * mult
                if ds in attention_level:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads_upsample)
                    )
                if level and i == num_res_blocks:
                    layers.append(UpSample(ch, conv_resample))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
            
        self.out = M.Sequential(
            M.GroupNorm(32, ch),
            SiLU(),
            M.Conv2d(model_channels, out_channels, 3, padding=1),
        )

        # zero module
        for p in self.out[-1].parameters():
            init.zeros_(p)

    def forward(self, x, timesteps, y=None):
        """Apply the model to an input batch.
        
        Args:
            x: an [N x C x ...] Tensor of inputs.
            timesteps: a 1-D batch of timesteps.
            y: an [N] Tensor of labels, if class-conditional.

        Return:
            an [N x C x ...] Tensor of outputs.
        """

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = F.concat([h, hs.pop()], axis=1)
            h = module(cat_in, emb)
        return self.out(h)

