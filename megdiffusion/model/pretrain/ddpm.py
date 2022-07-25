import megengine.hub as hub
from ..ddpm import UNet


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
    return UNet(
        total_timesteps=1000,
        in_resolution=32,
        in_channel=3,
        out_channel=3,
        base_channel=128,
        channel_multiplier=[1, 2, 2, 2],
        attention_resolutions=[16],
        num_res_blocks=2,
        dropout=0.1,
    )


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
    return UNet(
        total_timesteps=1000,
        in_resolution=32,
        in_channel=3,
        out_channel=3,
        base_channel=128,
        channel_multiplier=[1, 2, 2, 2],
        attention_resolutions=[16],
        num_res_blocks=2,
        dropout=0.1,
    )


@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_cifar10_converted.pkl")
def ddpm_cifar10_converted(**kwargs):
    """The pretrained DDPM model on CIFAR10 dataset, weights converted from original checkpoint.

    
    .. admonition:: Diffusion Schedule Configuration

       * Total timesteps: 1000
       * Beta schedule: linear
       * Model mean type: EPSILON
       * Model var type: FIXED_SMALL
       * Channel order: RGB

    """
    return UNet(
        total_timesteps=1000,
        in_resolution=32,
        in_channel=3,
        out_channel=3,
        base_channel=128,
        channel_multiplier=[1, 2, 2, 2],
        attention_resolutions=[16],
        num_res_blocks=2,
        dropout=0.1,
    )

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/ddpm_cifar10_ema_converted.pkl")
def ddpm_cifar10_ema_converted(**kwargs):
    """The pretrained DDPM model on CIFAR10 dataset, weights converted from original checkpoint.

    
    .. admonition:: Diffusion Schedule Configuration

       * Total timesteps: 1000
       * Beta schedule: linear
       * Model mean type: EPSILON
       * Model var type: FIXED_SMALL
       * Channel order: RGB

    """
    return UNet(
        total_timesteps=1000,
        in_resolution=32,
        in_channel=3,
        out_channel=3,
        base_channel=128,
        channel_multiplier=[1, 2, 2, 2],
        attention_resolutions=[16],
        num_res_blocks=2,
        dropout=0.1,
    )