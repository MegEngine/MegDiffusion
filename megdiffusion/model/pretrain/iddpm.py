import megengine.hub as hub
from ..iddpm import UNetModel

# Trained from scratch with MegEngine
retrained = [ 

]

# Converted from orginal paper provided checkpoints
converted = [
    "iddpm_cifar10_uncond_50M_500K_converted",
]

__all__ = retrained + converted

def _iddpm_unet_default_config(resolution: int, learn_sigma=False):
    if resolution == 32:
        channel_multiplier = [1, 2, 2, 2]
    elif resolution == 256:
        channel_multiplier = [1, 1, 2, 2, 4, 4]

    return UNetModel(
        in_channels=3,
        out_channels=6 if learn_sigma else 3,
        model_channels=128,
        channel_mult=channel_multiplier,
        num_res_blocks=3,
        attention_level=[2,4],
        dropout=0.3,
        use_scale_shift_norm=True,
        num_heads=4,
    )

def _iddpm_diffusion_default_config():
    return {
        "beta_schedule" : {
            "type": "cosine",
            "timesteps": 4000,
        },
        "model_mean_type": "EPSILON",
        "model_var_type": "LEARNED_RANGE",
        "rescale_timesteps": True,
    }

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/iddpm_cifar10_uncond_50M_500K_converted.pkl")
def iddpm_cifar10_uncond_50M_500K_converted(**kwargs):
    """The deault model configuration used in IDDPM paper on CIFAR10 dataset.
    Ported from: https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt
    """
    model = _iddpm_unet_default_config(resolution=32, learn_sigma=True)
    model.diffusion_config = _iddpm_diffusion_default_config()
    model.channel_order = "RGB"
    return model