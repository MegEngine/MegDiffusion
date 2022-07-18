"""Compare the forward ouput and autograd result between "the same module" in MegEngine and PyTorch.

It's usually used when you are porting a model from PyTorch to MegEngine (or reveresd case).
This script template can do the following things:

1. Create a Torch model(torch.module.Module) and saved all parameters in a dict;
2. Create a MegEngine model(megengine.module.Module) and load all parameters from the dict.
   
   If crashed, please check whether the model structure you ported is exactly the same as the original structure.
   The more common reason is the submodule names are inconsistent or need to reshape the shape of some parameters.

3. Now we have two model with same state, check the output result with the same input Tensor.

"""

import numpy as np

import torch
import megengine as mge

# import random
# seed = 37

# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True

# mge.random.seed(seed)
# mge.config.deterministic_kernel = True

# Examaple: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/nn.py
# Note for this example, the last layer wiil do zero initialize, we need remove it for comparing.
from torch_model import UNetModel

t_model = UNetModel(
    in_channels=3,
    model_channels=32,
    channel_mult=[1,2,2,2],
    out_channels=3,
    num_res_blocks=2,
    attention_resolutions=[16],
)

states = t_model.state_dict()
weights = {k: v.numpy() for k, v in states.items()}

from megdiffusion.model.iddpm import UNetModel

m_model = UNetModel(
    in_channels=3,
    model_channels=32,
    channel_mult=[1,2,2,2],
    out_channels=3,
    num_res_blocks=2,
    attention_resolutions=[16],
)

for k, v in weights.items():
    if k.endswith('bias') and "emb" not in k:  # Conv2d
         v = v.reshape(1, -1, 1, 1)
    if k.endswith('bias') and ("proj" in k or "qkv" in k):  # Conv1d
         v = v.reshape(1, -1, 1)
    weights[k] = v

m_model.load_state_dict(weights, strict=False)

batch_size = 4

x_torch = torch.randn(batch_size, 3, 32, 32)
t_torch = torch.randint(1000, (batch_size, ))
y_torch = t_model(x_torch, t_torch).cpu().detach().numpy()

x_mge = mge.Tensor(x_torch.cpu().detach().numpy())
t_mge = mge.Tensor(t_torch.cpu().detach().numpy())
y_mge = m_model(x_mge, t_mge).numpy()

np.testing.assert_allclose(y_torch, y_mge)