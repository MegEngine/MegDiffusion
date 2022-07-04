"""
MegEngine Hub Configuration for MegDiffusion. See :mod:`megengine.hub` for more details.

Here is an exmaple showing how to find and load `ddpm_cifar10` pre-trained model.

>>> megengine.hub.list("MegEngine/MegDiffusion")
>>> megengine.hub.help("MegEngine/MegDiffusion", "ddpm_cifar10")

>>> model = megengine.hub.load("MegEngine/MegDiffusion", "ddpm_cifar10", pretrained=True)
>>> model.eval()

Then you can use the pre-trained model to evaluate or do other things.
"""

from megdiffusion.model import *