import numpy as np

from megengine.module import Module

def ema(source: Module, target: Module, decay: float):
    """Update target module's parameters with Exponential Moving Average algorithm."""
    for sp, tp in zip(source.parameters(), target.parameters()):
        tp._reset(tp * decay + sp * (1 - decay))
        