"""
Paper: `Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2006.11239>`_ (a.k.a DDPM)

Experiment code implementation:

* The official Tensorflow implementation: https://github.com/hojonathanho/diffusion
* Unofficial PyTorch implementation (non-exhaustive list): 
  * https://github.com/lucidrains/denoising-diffusion-pytorch
  * https://github.com/w86763777/pytorch-ddpm
  * https://github.com/pesser/pytorch_diffusion

The model structure implemented here is as consistent as possible with the official implementation.
The differences between the code implementation and the paper description will be pointed out.

.. _conv-instead-of-nin:

In official Tensorflow implementation, default Tensor format is (N, H, W, C),
so the author design a ``nin()`` function to change shape from (N, H, W, in_C) to (N, H, W, out_C).
In MegEngine, default Tensor format is (N, C, H, W), so we can use Conv to change channel num:

    >>> M.Conv2d(in_ch, out_ch, 1, stride=1)
    shape from (N, in_ch, H, W) to (N, out_ch, H, W)

See ``nin()`` code: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py#L55

.. _resblock-with-attn:

The ``ResBlock`` is optional to add a ``AttnBlock`` at the end directly for constructing model easier.

.. _prefer-to-use-module:

Define ``M.Module`` instead of ``function`` can make it easier to organize model arch with ``M.Sequential``.
Even though there are no parameters in thoes module (such as nolinearity ``Swish``).
"""

from typing import List, Sequence

import math

import megengine.functional as F
import megengine.module as M
import megengine.module.init as init


class Swish(M.Module):  # nonlinearity
    """Element-wise :math:`x \times \frac {1}{1+\exp(-x)} `, i.e ``x * sigmoid(x)``.

    .. seealso::

       The original swish function has a const/trainable parameter :math:`\beta`.
       For :math:`\beta = 1`, it becomes equivalent to the Sigmoid-weighted Linear Unit.
       For more details, see: https://en.wikipedia.org/wiki/Swish_function
    """

    def forward(self, x):
        return F.sigmoid(x) * x

class GroupNorm(M.GroupNorm):
    def __init__(self, num_groups, num_channels):
        super().__init__(num_groups, num_channels, eps=1e-6)  # The default initial value in Tensorflow

class TimeEmbedding(M.Module):
    """Sinusoidal Positional Embedding with given timestep ``t`` information.

    .. seealso::

       Refer to tensorflow implementation and its comment:

       https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py#L90

       From `Fairseq SinusoidalPositionalEmbedding
       <https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py>`_.

       Build sinusoidal embeddings of any length.
       This matches the implementation in tensor2tensor, but differs slightly
       from the description in Section 3.5 of "Attention Is All You Need".
    """

    def __init__(self, total_timesteps, model_channels, time_embed_dim):
        super().__init__()
        emb = self._get_timestep_embedding(total_timesteps, model_channels)
        self.timembedding = M.Sequential(
            M.Embedding.from_pretrained(emb),
            M.Linear(model_channels, time_embed_dim),  # dense
            Swish(),
            M.Linear(time_embed_dim, time_embed_dim),  # dense
        )
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, M.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def _get_timestep_embedding(self, timesteps, embedding_dim):
        """Build sinusoidal embeddings, consider timesteps as num_embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = F.exp(F.arange(half_dim, dtype="float32") * -emb)
        pos = F.arange(timesteps, dtype="float32")  # discrete time step
        emb = pos[:, None] * emb[None, :]
        emb = F.concat([F.sin(emb), F.cos(emb)], axis=1).reshape(timesteps, -1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.concat([emb, F.zeros(timesteps, 1)], axis=1)
        return emb

    def forward(self, t):
        return self.timembedding(t)


class DownSample(M.Module):
    """"A downsampling layer with an optional convolution.

    Args:
        in_ch: channels in the inputs and outputs.
        use_conv: if ``True``, apply convolution to do downsampling; otherwise use pooling.
    """""

    def __init__(self, in_ch, with_conv=True):
        super().__init__()
        self.with_conv = with_conv

        if with_conv:
            self.main = M.Conv2d(in_ch, in_ch, 3, stride=2)  # Note not padding here
        else:
            self.main = M.AvgPool2d(2, stride=2)

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, M.Conv2d):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x, temb):  # add unused temb param here just for convince
        if self.with_conv:
            """In Tensorflow design, Conv2d's padding behavior is different.
            So we need do explicit padding here instead of giving Conv2d `padding` param.
            See https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_
            """
            x = F.nn.pad(x, [*[(0, 0) for i in range(x.ndim - 2)], (0, 1), (0, 1)])
        return self.main(x)

class UpSample(M.Module):
    """An upsampling layer with an optional convolution.

    Args:
        in_ch: channels in the inputs and outputs.
        use_conv: if ``True``, apply convolution after upsampling.
    """

    def __init__(self, in_ch, with_conv=True):
        super().__init__()
        if with_conv:
            self.main = M.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        else:
            self.main = M.Identity()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, M.Conv2d):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x, temb):  # add unused temb param here just for convince
        x = F.nn.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.main(x)


class AttnBlock(M.Module):
    """An attention block that allows spatial positions to attend to each other.

    Originally ported from here but use ``conv`` insead of ``nin``:

    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py#L66

    See :ref:`conv-instead-of-nin` for more details.
    """

    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = GroupNorm(32, in_ch)
        self.proj_q = M.Conv2d(in_ch, in_ch, 1, stride=1)
        self.proj_k = M.Conv2d(in_ch, in_ch, 1, stride=1)
        self.proj_v = M.Conv2d(in_ch, in_ch, 1, stride=1)
        self.proj = M.Conv2d(in_ch, in_ch, 1, stride=1)
        self._initialize()

    def _initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h).reshape(B, C, H*W)
        k = self.proj_k(h).reshape(B, C, H*W)
        v = self.proj_v(h).reshape(B, C, H*W)

        # einsum('b(hw)c,b(HW)c->b(hw)(HW)', q, k)
        w = q.transpose(0, 2, 1) @ k
        w = w * (int(C)**(-0.5))
        w = F.softmax(w, axis=-1)

        # Orignial: einsum('b(hw)(HW),b(HW)c->b(hw)c', w, v)  this is tf tensor format bhwc
        # Modified: einsum('bc(HW),b(hw)(HW)->bc(hw)', v, w)  megengine tensor format
        h = (v @ w.transpose(0, 2, 1)).reshape(B, C, H, W)

        h = self.proj(h)

        return x + h


class ResBlock(M.Module):
    """A residual block with timestep embedding, optional convolution short cut and attention.

    Args:
        in_channel: the number of input channels.
        out_channel: the number of output channels.
        time_embed_dim: the number of timestep embedding channels.
        dropout: the rate of dropout.
        ues_spatial_conv: Only valid when ``out_channel`` not equals to ``in_channel``,
            If ``True``, apply a spatial 3x3 kernel conv on shortcut to change channel num;
            Other wise, apply a smaller 1x1 kernel conv (without padding).
        use_attn: If ``True``, add an attention layer at the end.

    Note:

        * Arugument ``conv_shortcut`` is used in official DDPM Tensorflow code.
          When ``out_channel != out_channel``, apply ``conv`` or ``nin`` on shortcut.
          But we use name ``ues_spatial_conv`` here because we always use conv.
        * Arugument ``use_attn`` is not used in official DDPM Tensorflow code.
          We add it here to be more convenient when constructing UNet structure.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 time_embed_dim: int,
                 dropout: float,
                 use_attn: bool = False,
                 ues_spatial_conv: bool = False,
                 ):
        super().__init__()
        self.block1 = M.Sequential(
            GroupNorm(32, in_channel),
            Swish(),
            M.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
        )
        self.temb_proj = M.Sequential(   # add in timestep embedding
            Swish(),
            M.Linear(time_embed_dim, out_channel),
        )
        self.block2 = M.Sequential(
            GroupNorm(32, out_channel),
            Swish(),
            M.Dropout(dropout),
            M.Conv2d(out_channel, out_channel, 3, stride=1, padding=1)
        )

        if in_channel != out_channel:
            """The original tensorflow code use ``nin`` to change channel number:
            
            .. code-block:: python
            
               if conv_shortcut:
                   x = nn.conv2d(x, name='conv_shortcut', num_units=out_ch)
               else:
                   x = nn.nin(x, name='nin_shortcut', num_units=out_ch)
            
            We use 1x1 ``conv`` instead of ``nin`` here due to tensor format.
            See :ref:`conv-instead-of-nin` for more details.
            """
            if ues_spatial_conv:   # instead of ``conv_shortcut`` argument
                self.short_cut = M.Conv2d(
                    in_channel, out_channel, 3, stride=1, padding=1)
            else:
                self.short_cut = M.Conv2d(in_channel, out_channel, 1, stride=1)
        else:
            self.short_cut = M.Identity()

        if use_attn:
            self.attn = AttnBlock(out_channel)
        else:
            self.attn = M.Identity()

        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, (M.Conv2d, M.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h += self.short_cut(x)
        h = self.attn(h)
        return h


class UNet(M.Module):
    """The base model used in DDPM paper.From appendix B Experimental details:
    "Our neural network architecture follows the backbone of PixelCNN++, 
    which is a U-Net based on a Wide ResNet."

    Args:
        total_timesteps: the number of total timesteps, i.e ``T``, as a hyperparameter.
        in_resolution: resolution of input image with same height and width.
        in_channel: the number of input channels. E.g: ``3`` for RGB image.
        out_channel: the number of output channels. If `None`,
            the model predict noise by default, so equals to ``in_channel``.
        base_channel: base channel count for the model.
        channel_multiplier: channel multiplier for each level of the UNet.
            If also determine the total level num of the model.
        attention_resolutions: resolutions when use attention block.
        num_res_blocks: number of residual blocks per downsample.
        dropout: the rate of dropout.
        conv_resample: if ``True``, use learned convolutions for up/downsampling.
    """
    def __init__(self,
                 total_timesteps: int,
                 in_resolution: int,
                 in_channel: int,
                 out_channel: int = None,
                 base_channel: int = 128,
                 channel_multiplier: Sequence[float] = [1, 2, 4, 8],
                 attention_resolutions: Sequence[int] = [16],
                 num_res_blocks: int = 2,
                 dropout: float = 0,
                 conv_resample: bool =True,
        ):

        super().__init__()

        out_channel = in_channel if out_channel is None else out_channel

        # Timestep embedding
        time_embed_dim = base_channel * 4
        self.time_embedding = TimeEmbedding(
            total_timesteps, base_channel, time_embed_dim)

        self.head = M.Conv2d(in_channel, base_channel, 3, stride=1, padding=1)

        # record needed infomations to construct the completed model
        channels = [base_channel]
        cur_ch, cur_res = base_channel, in_resolution

        # Downsampling
        self.downblocks = []
        for level, mult in enumerate(channel_multiplier):
            out_ch = int(base_channel * mult)
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    cur_ch, out_ch, time_embed_dim, dropout, cur_res in attention_resolutions))
                cur_ch = out_ch
                channels.append(cur_ch)
            if level != len(channel_multiplier) - 1:
                self.downblocks.append(DownSample(cur_ch, with_conv=conv_resample))
                cur_res = cur_res / 2
                channels.append(cur_ch)

        # Middle
        self.middleblocks = [
            ResBlock(cur_ch, cur_ch, time_embed_dim, dropout, True),
            ResBlock(cur_ch, cur_ch, time_embed_dim, dropout, False),
        ]

        # Upsampling
        self.upblocks = []
        for level, mult in reversed(list(enumerate(channel_multiplier))):
            out_ch = int(base_channel * mult)
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    channels.pop() + cur_ch, out_ch, time_embed_dim, dropout, cur_res in attention_resolutions
                ))
                cur_ch = out_ch
            if level != 0:
                cur_res = cur_res * 2
                self.upblocks.append(UpSample(cur_ch, with_conv=conv_resample))
        assert len(channels) == 0

        self.tail = M.Sequential(
            GroupNorm(32, cur_ch),
            Swish(),
            M.Conv2d(cur_ch, out_channel, 3, stride=1, padding=1)
        )

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):

        # Timestep embedding
        temb = self.time_embedding(t)

        h = self.head(x)
        concat_list = [h]  # Storage feature maps for skip connection

        # Downsampling
        for layer in self.downblocks:
            h = layer(h, temb)  # temb is not accessed in downsample block
            concat_list.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)

        # Upsampling        
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = F.concat([h, concat_list.pop()], axis=1)  # skip connection
            h = layer(h, temb)  # temb is not accessed in upsample block

        h = self.tail(h)

        assert not concat_list
        return h
