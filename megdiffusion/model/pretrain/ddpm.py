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
    "ddpm_celebahq_256_converted",
    "ddpm_celebahq_256_ema_converted",
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

def _ddpm_diffusion_default_config():
    return {
        "beta_schedule" : {
            "type": "linear",
            "timesteps": 1000,
            "start": 0.0001,
            "end": 0.02
        },
        "model_mean_type": "EPSILON",
        "model_var_type": "FIXED_SMALL",
    }

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_cifar10.pkl")
def ddpm_cifar10(**kwargs):
    """The pretrained DDPM model on CIFAR10 dataset."""
    model = _ddpm_unet_default_config(resolution=32)
    model.diffusion_config = _ddpm_diffusion_default_config()
    model.channel_order = "BGR"
    return model


@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_cifar10_ema.pkl")
def ddpm_cifar10_ema(**kwargs):
    """The pretrained DDPM model on CIFAR10 dataset (with EMA)."""
    model = _ddpm_unet_default_config(resolution=32)
    model.diffusion_config = _ddpm_diffusion_default_config()
    model.channel_order = "BGR"
    return model


@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_cifar10_converted.pkl")
def ddpm_cifar10_converted(**kwargs):
    """The pretrained DDPM model on CIFAR10 dataset, which is converted from original checkpoint."""
    model = _ddpm_unet_default_config(resolution=32)
    model.diffusion_config = _ddpm_diffusion_default_config()
    model.channel_order = "RGB"
    return model

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_cifar10_ema_converted.pkl")
def ddpm_cifar10_ema_converted(**kwargs):
    """The pretrained DDPM model on CIFAR10 dataset, which is converted from original checkpoint."""
    model = _ddpm_unet_default_config(resolution=32)
    model.diffusion_config = _ddpm_diffusion_default_config()
    model.channel_order = "RGB"
    return model

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_lsun_bedroom_converted.pkl")
def ddpm_lsun_bedroom_converted(**kwargs):
    """The pretrained DDPM model on LSUN cat dataset, which is converted from original checkpoint."""
    model = _ddpm_unet_default_config(resolution=256)
    model.diffusion_config = _ddpm_diffusion_default_config()
    model.channel_order = "RGB"
    return model

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_lsun_bedroom_ema_converted.pkl")
def ddpm_lsun_bedroom_ema_converted(**kwargs):
    """The pretrained DDPM model on LSUN cat dataset, which is converted from original checkpoint."""
    model = _ddpm_unet_default_config(resolution=256)
    model.diffusion_config = _ddpm_diffusion_default_config()
    model.channel_order = "RGB"
    return model

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_lsun_cat_converted.pkl")
def ddpm_lsun_cat_converted(**kwargs):
    """The pretrained DDPM model on LSUN cat dataset, which is converted from original checkpoint."""
    model = _ddpm_unet_default_config(resolution=256)
    model.diffusion_config = _ddpm_diffusion_default_config()
    model.channel_order = "RGB"
    return model

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_lsun_cat_ema_converted.pkl")
def ddpm_lsun_cat_ema_converted(**kwargs):
    """The pretrained DDPM model on LSUN cat dataset, which is converted from original checkpoint."""
    model = _ddpm_unet_default_config(resolution=256)
    model.diffusion_config = _ddpm_diffusion_default_config()
    model.channel_order = "RGB"
    return model

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_lsun_church_converted.pkl")
def ddpm_lsun_church_converted(**kwargs):
    """The pretrained DDPM model on LSUN cat dataset, which is converted from original checkpoint."""
    model = _ddpm_unet_default_config(resolution=256)
    model.diffusion_config = _ddpm_diffusion_default_config()
    model.channel_order = "RGB"
    return model

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_lsun_church_ema_converted.pkl")
def ddpm_lsun_church_ema_converted(**kwargs):
    """The pretrained DDPM model on LSUN cat dataset, which is converted from original checkpoint."""
    model = _ddpm_unet_default_config(resolution=256)
    model.diffusion_config = _ddpm_diffusion_default_config()
    model.channel_order = "RGB"
    return model

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_celebahq_256_converted.pkl")
def ddpm_celebahq_256_converted(**kwargs):
    """The pretrained DDPM model on CelebaHQ-256 dataset, which is converted from original checkpoint."""
    model = _ddpm_unet_default_config(resolution=256)
    model.diffusion_config = _ddpm_diffusion_default_config()
    model.channel_order = "RGB"
    return model

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_celebahq_256_ema_converted.pkl")
def ddpm_celebahq_256_ema_converted(**kwargs):
    """The pretrained DDPM model on CelebaHQ-256 dataset, which is converted from original checkpoint."""
    model = _ddpm_unet_default_config(resolution=256)
    model.diffusion_config = _ddpm_diffusion_default_config()
    model.channel_order = "RGB"
    return model
