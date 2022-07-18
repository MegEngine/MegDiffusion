import numpy as np

import megengine as mge

from megdiffusion.model.ddpm import UNet

def test_model_ddpm_cifar10():
    batch_size, channel_szie, resolution_siize = 4, 3, 32
    model = UNet(
        total_timesteps=1000,
        in_resolution=resolution_siize,
        in_channel=channel_szie,
    )
    x = mge.random.normal(0, 1, (batch_size, channel_szie, resolution_siize, resolution_siize))
    t = mge.Tensor(np.random.randint(0, 1000, (batch_size, )))
    y = model(x, t)

from megdiffusion.model.iddpm import UNetModel

def test_model_iddpm_cifar10():
    batch_size, channel_szie, resolution_siize = 4, 3, 32
    model = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[16],
    )
    x = mge.random.normal(0, 1, (batch_size, channel_szie, resolution_siize, resolution_siize))
    t = mge.Tensor(np.random.randint(0, 1000, (batch_size, )))
    y = model(x, t)