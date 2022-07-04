import numpy as np

import megengine as mge
import megengine.functional as F

from tqdm import tqdm

from ..utils.schedule import linear_schedule
from ..utils import batch_broadcast

class GaussionDiffusion:

    def __init__(self, timesteps, model, betas = None) -> None:

        self.timesteps = timesteps
        self.model = model

        # pre-calculate some constant values in NumpPy float64 dtype to keep accuracy.
        # then copy these values from host to device in MegEngine float32 dtype
        # because float64 is not supported in MegEngine right now :(
        def host2device(data):
            return mge.Tensor(data, dtype="float32")
        
        # define beta schedule
        betas = linear_schedule(timesteps) if betas is None else betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        # define alphas and alphas_cumprod
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.)
        sqrt_recip_alphas = np.sqrt(1. / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = np.log(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) /
            (1. - alphas_cumprod)
        )

        # copy and store these values on GPU device (if exists) in advance
        self.betas = host2device(betas)
        self.alphas = host2device(alphas)
        self.alphas_cumprod = host2device(alphas_cumprod)
        self.alphas_cumprod_prev = host2device(alphas_cumprod_prev)
        self.alphas_cumprod_next = host2device(alphas_cumprod_next)
        self.sqrt_recip_alphas = host2device(sqrt_recip_alphas)
        self.sqrt_alphas_cumprod = host2device(sqrt_alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = host2device(sqrt_one_minus_alphas_cumprod)
        self.log_one_minus_alphas_cumprod = host2device(log_one_minus_alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = host2device(sqrt_recip_alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = host2device(sqrt_recipm1_alphas_cumprod)
        self.posterior_variance = host2device(posterior_variance)
        self.posterior_log_variance_clipped = host2device(posterior_log_variance_clipped)
        self.posterior_mean_coef1 = host2device(posterior_mean_coef1)
        self.posterior_mean_coef2 = host2device(posterior_mean_coef2)

    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0) using reparameterization trick."""
        shape = x_start.shape
        noise = mge.random.normal(0, 1, shape) if noise is None else noise

        mean = batch_broadcast(self.sqrt_alphas_cumprod[t], shape) * x_start
        std = batch_broadcast(self.sqrt_one_minus_alphas_cumprod[t], shape)
        return mean + std * noise

    def p_sample(self, x, t, clip_denoised=False):
        """Sample from p_{theta} (x_{t-1} | x_t) using reparameterization trick."""

        shape = x.shape

        # if t == 0, the sample do not need to be denoised, so add a mask here
        nozero_mask = batch_broadcast(t != 0, shape)
        noise = nozero_mask * mge.random.normal(0, 1, shape)

        model_output = self.model(x, t)  # predict t level noise in this case

        predict_x_start = (
            batch_broadcast(self.sqrt_recip_alphas_cumprod[t], shape) * x -
            batch_broadcast(self.sqrt_recipm1_alphas_cumprod[t], shape) * model_output
        )

        # All the image values are scaled to [-1, 1], so clip them here
        if clip_denoised:
            predict_x_start = F.clip(predict_x_start, -1., 1.)

        model_mean = (
            batch_broadcast(self.posterior_mean_coef1[t], shape) * predict_x_start
            + batch_broadcast(self.posterior_mean_coef2[t], shape) * x
        )

        model_var = batch_broadcast(self.posterior_log_variance_clipped[t], shape)  # Fixed

        x_denoised = model_mean + F.exp(0.5 * model_var) * noise

        return x_denoised

    def p_sample_loop(self, shape):
        x = mge.random.normal(0, 1, shape)
        for i in tqdm(reversed(range(0, self.timesteps)), 
            desc="Generating image from noise", total=self.timesteps):
            x = self.p_sample(x, F.full((shape[0],), i))
        return x
        
    def p_loss(self, x_start, t=None, noise=None):
        if t is None:
            t = mge.Tensor(np.random.randint(0, self.timesteps, len(x_start)))
        noise = mge.random.normal(0, 1, x_start.shape) if noise is None else noise

        x_noisy = self.q_sample(x_start, t, noise)
        predict_noise = self.model(x_noisy, t)

        loss = F.nn.square_loss(noise, predict_noise)

        return loss