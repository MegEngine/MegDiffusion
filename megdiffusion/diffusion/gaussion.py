"""Gaussion Diffusion Scheduler.

Modified from OpenAI improved/guided diffusion codebase:
https://github.com/openai/guided-diffusion/blob/master/guided_diffusion/gaussian_diffusion.py#L328

OpenAI's code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
"""

import numpy as np

import megengine as mge
import megengine.functional as F

from tqdm import tqdm

from .schedule import linear_schedule
from ..utils import batch_broadcast

class GaussionDiffusion:

    def __init__(
        self,
        *,
        timesteps = 1000,
        betas = None,
        model = None, 
        model_mean_type = "EPSILON",
        model_var_type = "FIXED_SMALL",
        rescale_timesteps = False,
    ) -> None:

        assert model_mean_type in ["PREVIOUS_X", "START_X", "EPSILON"]
        assert model_var_type in ["FIXED_SMALL", "FIXED_LARGE", "LEARNED", "LEARNED_RANGE"]

        self.timesteps = timesteps
        self.model = model
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.rescale_timesteps = rescale_timesteps

        # define beta schedule
        self.betas = linear_schedule(timesteps) if betas is None else betas
        self._pre_calculate(self.betas)

    def _pre_calculate(self, betas):
        """Pre-calculate constant values frequently used in formulas appears in paper.
        Calculated values will be copied to GPU (if it's default device) in advance.
        It can prevent lots of copy operations in subsequent processes.
        
        Args:
            betas: a 1-D np.array including scheduled beta values.
        """

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
        frac_coef1_coef2 = posterior_mean_coef1 / posterior_mean_coef2

        def host2device(data):
            return mge.Tensor(data, dtype="float32")

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
        self.frac_coef1_coef2 = host2device(frac_coef1_coef2)

    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0) using reparameterization trick."""
        shape = x_start.shape
        noise = mge.random.normal(0, 1, shape) if noise is None else noise

        mean = batch_broadcast(self.sqrt_alphas_cumprod[t], shape) * x_start
        std = batch_broadcast(self.sqrt_one_minus_alphas_cumprod[t], shape)
        return mean + std * noise

    def p_sample(self, x_t, t, clip_denoised=False):
        """Sample from p_{theta} (x_{t-1} | x_t) using reparameterization trick."""
        shape = x_t.shape

        # if t == 0, the sample do not need to be denoised, so add a mask here
        nozero_mask = batch_broadcast(t != 0, shape)
        noise = nozero_mask * mge.random.normal(0, 1, shape)

        model_output = self.model(
            x_t, 
            t * 1000.0 / self.timesteps if self.rescale_timesteps else t
        )

        # handle with model_output according to the variance type (fixed or learned)
        if self.model_var_type == "FIXED_SMALL":
            model_log_var = batch_broadcast(self.posterior_log_variance_clipped[t], shape)
        elif self.model_var_type == "FIXED_LARGE":
            model_log_var = batch_broadcast(
                F.concat((self.posterior_log_variance_clipped[1], self.betas[1:]), axis=1),
                shape,
            )
        else:  # model's output contains learned variance value (the 2nd 3 channels)
            model_output, model_var_values = F.split(model_output, 2, axis=1)
            if self.model_var_type == "LEARNED":  # learned variance directly
                model_log_var = model_var_values
            elif self.model_var_type == "LEARNED_RANGE":  # IDDPM Eq. (15)
                min_log = batch_broadcast(self.posterior_log_variance_clipped[t], shape)
                max_log = batch_broadcast(F.log(self.betas[t]), shape)
                # The model_var_values is [-1, 1] and should convert to [0, 1].
                frac = (model_var_values + 1) / 2
                model_log_var = frac * max_log + (1 - frac) * min_log

        if self.model_mean_type == "PREVIOUS_X":  # model_ouput is x_{t-1}
            predict_x_start = (  # formula x_0 = (x_{t-1} - coef2 * x_t) / coef1, not mentioned in papaer
                batch_broadcast(1.0 / self.posterior_mean_coef1[t], shape) * model_output -
                batch_broadcast(self.frac_coef1_coef2[t], shape) * x_t
            )
        elif self.model_mean_type == "EPSILON":  # model_output is noise between x_{t-1} and x_{t}
            predict_x_start = (
                batch_broadcast(self.sqrt_recip_alphas_cumprod[t], shape) * x_t -
                batch_broadcast(self.sqrt_recipm1_alphas_cumprod[t], shape) * model_output
            )
        else:  # model_output is x_0 directly
            predict_x_start = model_output

        # All the image values are scaled to [-1, 1], so clip them here
        if clip_denoised:
            predict_x_start = F.clip(predict_x_start, -1., 1.)

        # get predicted x_{t-1} from predicted x_0 and input x_t
        model_mean = (
            batch_broadcast(self.posterior_mean_coef1[t], shape) * predict_x_start
            + batch_broadcast(self.posterior_mean_coef2[t], shape) * x_t
        )
        
        return model_mean + F.exp(0.5 * model_log_var) * noise

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