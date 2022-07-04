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