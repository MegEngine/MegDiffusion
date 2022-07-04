# MegDiffusion

MegEngine implementation of Diffusion Models (in early development).

Current maintainer: [@MegChai](https://github.com/MegChai)

## Usage
### Infer with pre-trained models

Now users can use `megengine.hub` to get pre-trained models directly:

```python
megengine.hub.list("MegEngine/MegDiffusion:main")
megengine.hub.help("MegEngine/MegDiffusion:main", "ddpm_cifar10")
model = megengine.hub.load("MegEngine/MegDiffusion:main", "ddpm_cifar10", pretrained=True)
model.eval()
```

Or if you have downloaded or installed MegDiffusion, you can get pre-trained models from `model` module.

```python
from megdiffusion.model import ddpm_cifar10
model = ddpm_cifar10(pretrained=True)
model.eval()
```

The inference [script](megdiffusion/scripts/inference.py) shows how to generate 64 CIFAR10-like images and make a grid of them:

```shell
python3 -m megdiffusion.scripts.inference
```

### Train from scratch

- Take DDPM CIFAR10 for example:

  ```shell
  python3 -m megdiffusion.scripts.train \
      --flagfile ./megdiffusion/config/ddpm-cifar10.txt
  ```

- [Optional] Overwrite arguments:

   ```shell
   python3 -m megdiffusion.scripts.train \
      --flagfile ./megdiffusion/config/ddpm-cifar10.txt \
      --logdir ./path/to/logdir \
      --batch_size=64 \
      --save_step=100000 \
      --parallel=True
   ```

Known issues:
- Training with single GPU & using gradient clipping will cause error in MegEngine 1.9.x version.

## Development

```shell
python3 -m pip install -r requirements.txt
python3 -m pip install -v -e .
```

Develop this project with a new branch locally, remember to add necessary test codes.
If finished, submit Pull Request to the `main` branch then just wait for review.

## Acknowledgment

The following open-sourced projects was referenced here:

- [hojonathanho](https://github.com/hojonathanho)/[diffusion](https://github.com/hojonathanho/diffusion): The official Tensorflow implementation of DDPM.
- [w86763777](https://github.com/w86763777)/[pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm): Unofficial PyTorch implementation of Denoising Diffusion Probabilistic Models.

Thanks to people including [@gaohuazuo](https://github.com/gaohuazuo), [@xxr3376](https://github.com/xxr3376), [@P2Oileen](https://github.com/P2Oileen) and other contributors for support in this project. The R&D platform and the resources required for the experiment are provided by [MEGVII](https://megvii.com/) Inc. The deep learning framework used in this project is [MegEngine](https://github.com/MegEngine/MegEngine) -- a magic weapon.

## Citations

```
@article{ho2020denoising,
    title   = {Denoising Diffusion Probabilistic Models},
    author  = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year    = {2020},
    eprint  = {2006.11239},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```