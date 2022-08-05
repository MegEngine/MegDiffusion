"""
Convert improved diffusion Pytorch pre-trained model checkpoints to MegEngine.

Please install improved-diffusion first:

.. code-block:: shell

   git clone git@github.com:openai/improved-diffusion.git
   python3 -m pip install -e improved-diffusion

Scripts torch model and diffusion build part is modified from:

https://github.com/openai/improved-diffusion/blob/main/scripts/image_sample.py
"""

import argparse
import pdb
import numpy as np
import os
import random
import yaml

import torch
from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import megengine as mge
from megdiffusion.model.iddpm import UNetModel

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser(description="Convert pretrained model checkpoints to MegEngine.")
    parser.add_argument("--config", type=str, required=True, help="The path to the config file including MegEngine model information.")
    parser.add_argument("--output_dir", type=str, required=True, help="The path to saved the converted checkpoint.")
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    
    args = create_argparser().parse_args()
    
    # Load torch model and init megengine model
    torch_model, torch_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    torch_model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    print("Loaded model from {}".format(args.model_path))

    original_name = os.path.splitext(os.path.basename(args.model_path))[0]
    target_filename = "iddpm_" + original_name + "_converted.pkl"
    target_path = os.path.join(args.output_dir, target_filename)
    
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    mge_model = UNetModel(**config["model"])

    # Convert
    states = torch_model.state_dict()
    weights = {k: v.numpy() for k, v in states.items()}

    for k, v in weights.items():
        if k.endswith("bias") and "emb" not in k:  # Conv2d (exclude embedding)
            v = v.reshape(1, -1, 1, 1)
        if k.endswith("bias") and ("proj" in k or "qkv" in k):  # Conv1d
            v = v.reshape(1, -1, 1)
        weights[k] = v
 
    mge_model.load_state_dict(weights)

    img_resolution = config["data"]["img_resolution"]
    img_size = config["data"]["image_size"]

    # Compare
    torch_model.eval()
    x_torch = torch.randn(4, img_resolution, img_size, img_size)
    t_torch = torch.randint(1000, (4,))
    y_torch = torch_model(x_torch, t_torch).cpu().detach().numpy()

    mge_model.eval()
    x_mge = mge.Tensor(x_torch.cpu().detach().numpy())
    t_mge = mge.Tensor(t_torch.cpu().detach().numpy())
    y_mge = mge_model(x_mge, t_mge).numpy()

    try:
        np.testing.assert_allclose(y_torch, y_mge, atol=1e-5)
    except AssertionError as e:
        print("Output Error: {}".format(e))

    # Save
    mge.save(mge_model.state_dict(), target_path)
    print("Saved converted model to {}".format(target_path))


if __name__ == "__main__":
    main()