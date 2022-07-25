import megengine.hub as hub
from ..ddpm import UNet

# Trained from scratch with MegEngine
retrained = [ 
    "ddpm_cifar10",
    "ddpm_cifar10_ema",
]

# Converted from orginal paper provided checkpoints
converted = [
    "ddpm_cifar10_converted",
    "ddpm_cifar10_ema_converted",
    "ddpm_lsun_bedroom_converted",
    "ddpm_lsun_bedroom_ema_converted",
    "ddpm_lsun_cat_converted",
    "ddpm_lsun_cat_ema_converted",
    "ddpm_lsun_church_converted",
    "ddpm_lsun_church_ema_converted",
]

__all__ = retrained + converted

def _ddpm_unet_default_config(resolution: int):
    if resolution == 32:
        channel_multiplier = [1, 2, 2, 2]
    elif resolution == 256:
        channel_multiplier = [1, 1, 2, 2, 4, 4]

    return UNet(
        total_timesteps=1000,
        in_resolution=resolution,
        in_channel=3,
        out_channel=3,
        base_channel=128,
        channel_multiplier=channel_multiplier,
        attention_resolutions=[16],
        num_res_blocks=2,
        dropout=0.1,
    )


@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_cifar10.pkl")
def ddpm_cifar10(**kwargs):
    """The pretrained DDPM model on CIFAR10 dataset.
    
    .. admonition:: Diffusion Schedule Configuration

       * Total timesteps: 1000
       * Beta schedule name: linear
       * Model mean type: EPSILON
       * Model var type: FIXED_SMALL
       * Channel order: BGR

    """
    return _ddpm_unet_default_config(resolution=32)


@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_cifar10_ema.pkl")
def ddpm_cifar10_ema(**kwargs):
    """The pretrained DDPM model on CIFAR10 dataset (with EMA).
    
    .. admonition:: Diffusion Schedule Configuration

       * Total timesteps: 1000
       * Beta schedule name: linear
       * Model mean type: EPSILON
       * Model var type: FIXED_SMALL
       * Channel order: BGR

    """
    return _ddpm_unet_default_config(resolution=32)


@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_cifar10_converted.pkl")
def ddpm_cifar10_converted(**kwargs):
    """The pretrained DDPM model on CIFAR10 dataset, which is converted from original checkpoint.

    
    .. admonition:: Diffusion Schedule Configuration

       * Total timesteps: 1000
       * Beta schedule: linear
       * Model mean type: EPSILON
       * Model var type: FIXED_SMALL
       * Channel order: RGB

    """
    return _ddpm_unet_default_config(resolution=32)

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_cifar10_ema_converted.pkl")
def ddpm_cifar10_ema_converted(**kwargs):
    """The pretrained DDPM model on CIFAR10 dataset, which is converted from original checkpoint.

    
    .. admonition:: Diffusion Schedule Configuration

       * Total timesteps: 1000
       * Beta schedule: linear
       * Model mean type: EPSILON
       * Model var type: FIXED_SMALL
       * Channel order: RGB

    """
    return _ddpm_unet_default_config(resolution=32)

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_lsun_bedroom_converted.pkl")
def ddpm_lsun_bedroom_converted(**kwargs):
    """The pretrained DDPM model on LSUN cat dataset, which is converted from original checkpoint.

    
    .. admonition:: Diffusion Schedule Configuration

       * Total timesteps: 1000
       * Beta schedule: linear
       * Model mean type: EPSILON
       * Model var type: FIXED_SMALL
       * Channel order: RGB

    """
    return _ddpm_unet_default_config(resolution=256)

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_lsun_bedroom_ema_converted.pkl")
def ddpm_lsun_bedroom_ema_converted(**kwargs):
    """The pretrained DDPM model on LSUN cat dataset, which is converted from original checkpoint.

    
    .. admonition:: Diffusion Schedule Configuration

       * Total timesteps: 1000
       * Beta schedule: linear
       * Model mean type: EPSILON
       * Model var type: FIXED_SMALL
       * Channel order: RGB

    """
    return _ddpm_unet_default_config(resolution=256)

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_lsun_cat_converted.pkl")
def ddpm_lsun_cat_converted(**kwargs):
    """The pretrained DDPM model on LSUN cat dataset, which is converted from original checkpoint.

    
    .. admonition:: Diffusion Schedule Configuration

       * Total timesteps: 1000
       * Beta schedule: linear
       * Model mean type: EPSILON
       * Model var type: FIXED_SMALL
       * Channel order: RGB

    """
    return _ddpm_unet_default_config(resolution=256)

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_lsun_cat_ema_converted.pkl")
def ddpm_lsun_cat_ema_converted(**kwargs):
    """The pretrained DDPM model on LSUN cat dataset, which is converted from original checkpoint.

    
    .. admonition:: Diffusion Schedule Configuration

       * Total timesteps: 1000
       * Beta schedule: linear
       * Model mean type: EPSILON
       * Model var type: FIXED_SMALL
       * Channel order: RGB

    """
    return _ddpm_unet_default_config(resolution=256)

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_lsun_church_converted.pkl")
def ddpm_lsun_church_converted(**kwargs):
    """The pretrained DDPM model on LSUN cat dataset, which is converted from original checkpoint.

    
    .. admonition:: Diffusion Schedule Configuration

       * Total timesteps: 1000
       * Beta schedule: linear
       * Model mean type: EPSILON
       * Model var type: FIXED_SMALL
       * Channel order: RGB

    """
    return _ddpm_unet_default_config(resolution=256)

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_lsun_church_ema_converted.pkl")
def ddpm_lsun_church_ema_converted(**kwargs):
    """The pretrained DDPM model on LSUN cat dataset, which is converted from original checkpoint.

    
    .. admonition:: Diffusion Schedule Configuration

       * Total timesteps: 1000
       * Beta schedule: linear
       * Model mean type: EPSILON
       * Model var type: FIXED_SMALL
       * Channel order: RGB

    """
    return _ddpm_unet_default_config(resolution=256)
