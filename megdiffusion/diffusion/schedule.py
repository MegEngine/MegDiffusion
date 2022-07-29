import math
import numpy as np

def linear_schedule(timesteps, start=0.0001, end=0.02, range=1000):
    """Linear schedule from Ho et al, extended to work for any number of diffusion steps."""
    scale = range / timesteps
    return np.linspace(scale * start, scale * end, timesteps, dtype=np.float64)

def cosine_schedule(timesteps, max_beta=0.999):
    """Proposed in Improved Denoising Diffusion Probabilistic Models"""
    return _betas_for_alpha_bar(
        timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        max_beta
    )

def _betas_for_alpha_bar(timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given ``alpha_t_bar`` function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    Ported from: 
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

    Args:
        timesteps: the number of betas to produce.
        alpha_bar: a lambda that takes an argument t from 0 to 1 and
            produces the cumulative product of (1-beta) up to that
            part of the diffusion process.
        max_beta: the maximum beta to use; use values lower than 1 to
            prevent singularities.
    """
    betas = []
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)

def build_beta_schedule(type: str, timesteps: int, **kwargs):

    mapping = {
        "linear": linear_schedule,
        "cosine": cosine_schedule,
    }

    beta_schedule = mapping[type.lower()]

    return beta_schedule(timesteps=timesteps, **kwargs)