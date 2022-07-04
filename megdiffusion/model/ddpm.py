from typing import List, Sequence

import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import megengine.module.init as init


class Swish(M.Module):
    def forward(self, x):
        return F.sigmoid(x) * x


class TimeEmbedding(M.Module):
    """Sinusoidal Positional Embedding with given timestep t"""

    def __init__(self, total_timesteps, model_channels, time_embed_dim):
        assert model_channels % 2 == 0
        super().__init__()
        emb = F.arange(0, model_channels, step=2) / \
            model_channels * F.log(10000)
        emb = F.exp(-emb)
        pos = F.arange(total_timesteps, dtype="float32")
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [total_timesteps, model_channels // 2]
        emb = F.stack([F.sin(emb), F.cos(emb)], axis=-1)
        assert list(emb.shape) == [total_timesteps, model_channels // 2, 2]
        emb = emb.reshape(total_timesteps, model_channels)

        self.timembedding = M.Sequential(
            M.Embedding.from_pretrained(emb),
            M.Linear(model_channels, time_embed_dim),
            Swish(),
            M.Linear(time_embed_dim, time_embed_dim),
        )
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, M.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        return self.timembedding(t)


class DownSample(M.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = M.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        return self.main(x)


class UpSample(M.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = M.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = F.nn.interpolate(x, scale_factor=2, mode="nearest")
        return self.main(x)


class AttnBlock(M.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = M.GroupNorm(32, in_ch)
        self.proj_q = M.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = M.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = M.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = M.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self._initialize()

    def _initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.transpose(0, 2, 3, 1).reshape(B, H*W, C)
        k = k.reshape(B, C, H*W)
        w = q @ k
        assert list(w.shape) == [B, H*W, H*W]
        w = w * (int(C)**(-0.5))
        w = F.softmax(w, axis=-1)

        v = v.transpose(0, 2, 3, 1).reshape(B, H*W, C)
        h = w @ v
        assert list(h.shape) == [B, H*W, C]
        h = h.reshape(B, H, W, C).transpose(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(M.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 time_embed_dim: int,
                 dropout: float,
                 use_attn: bool = False
                 ):
        super().__init__()
        self.block1 = M.Sequential(
            M.GroupNorm(32, in_channel),
            Swish(),
            M.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
        )
        self.temb_proj = M.Sequential(
            Swish(),
            M.Linear(time_embed_dim, out_channel),
        )
        self.block2 = M.Sequential(
            M.GroupNorm(32, out_channel),
            Swish(),
            M.Dropout(dropout),
            M.Conv2d(out_channel, out_channel, 3, stride=1, padding=1)
        )

        if in_channel != out_channel:
            self.short_cut = M.Conv2d(
                in_channel, out_channel, 1, stride=1, padding=0)
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
    """The UNet model used in DDPM paper."""

    def __init__(self,
                 total_timesteps: int,
                 in_resolution: int,
                 in_channel: int,
                 out_channel: int = None,
                 base_channel: int = 128,
                 chanel_multiplier: Sequence[int] = [1, 2, 2, 2],
                 attention_resolutions: Sequence[int] = [16],
                 num_res_blocks: int = 2,
                 dropout: float = 0.1,
                 ):

        super().__init__()

        out_channel = in_channel if out_channel is None else out_channel
        time_embed_dim = base_channel * 4
        self.time_embedding = TimeEmbedding(
            total_timesteps, base_channel, time_embed_dim)

        self.head = M.Conv2d(in_channel, base_channel, 3, stride=1, padding=1)

        channels = [base_channel]
        cur_ch, cur_res = base_channel, in_resolution

        self.downblocks = []
        for i, mult in enumerate(chanel_multiplier):
            out_ch = base_channel * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    cur_ch, out_ch, time_embed_dim, dropout, cur_res in attention_resolutions))
                cur_ch = out_ch
                channels.append(cur_ch)
            if i != len(chanel_multiplier) - 1:
                self.downblocks.append(DownSample(cur_ch))
                cur_res = cur_res / 2
                channels.append(cur_ch)

        self.middleblocks = [
            ResBlock(cur_ch, cur_ch, time_embed_dim, dropout, True),
            ResBlock(cur_ch, cur_ch, time_embed_dim, dropout, False),
        ]

        self.upblocks = []
        for i, mult in reversed(list(enumerate(chanel_multiplier))):
            out_ch = base_channel * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    channels.pop() + cur_ch, out_ch, time_embed_dim, dropout, cur_res in attention_resolutions
                ))
                cur_ch = out_ch
            if i != 0:
                cur_res = cur_res * 2
                self.upblocks.append(UpSample(cur_ch))
        assert len(channels) == 0

        self.tail = M.Sequential(
            M.GroupNorm(32, cur_ch),
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
        temb = self.time_embedding(t)
        
        h = self.head(x)
        concat_list = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            concat_list.append(h)
        for layer in self.middleblocks:
            h = layer(h, temb)
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = F.concat([h, concat_list.pop()], axis=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(concat_list) == 0
        return h


@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_cifar10.pkl")
def ddpm_cifar10(**kwargs):
    """The deault model configuration used in DDPM paper on CIFAR10 dataset."""
    return UNet(
        total_timesteps = 1000,
        in_resolution = 32,
        in_channel = 3,
        out_channel = 3,
        base_channel = 128,
        chanel_multiplier = [1, 2, 2, 2],
        attention_resolutions = [16],
        num_res_blocks = 2,
        dropout = 0.1,
    )