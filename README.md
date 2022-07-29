# MegDiffusion

MegEngine implementation of Diffusion Models (in early development).

Current maintainer: [@MegChai](https://github.com/MegChai)

## Usage
### Infer with pre-trained models

Now users can use `megengine.hub` to get pre-trained models directly:

```python
import megengine

repo_info = "MegEngine/MegDiffusion:main"
megengine.hub.list(repo_info)

preatrained_model = "ddpm_cifar10_ema_converted"
megengine.hub.help(repo_info, preatrained_model)

model = megengine.hub.load(repo_info, preatrained_model, pretrained=True)
model.eval()
```

Note that using `megengine.hub` will download the whole repository from it's host or using cache.

If you have downloaded or installed MegDiffusion, you can get pre-trained models from `pretrain` module.

```python
from megdiffusion.model import pretrain

model = pretrain.ddpm_cifar10_ema_converted(pretrained=True)
model.eval()
```

The sample [script](megdiffusion/pipeline/ddpm/sample.py) shows how to generate 64 CIFAR10-like images and make a grid of them:

```shell
python3 -m megdiffusion.pipeline.ddpm.sample
```

### Train from scratch

- Take DDPM CIFAR10 for example:

  ```shell
  python3 -m megdiffusion.pipeline.ddpm.train \
      --config ./configs/ddpm/cifar10.yaml
  ```

- [Optional] Overwrite arguments:

  ```shell
  python3 -m megdiffusion.pipeline.ddpm.train \
     --config ./configs/ddpm/cifar10.yaml \
     --logdir ./path/to/logdir \
     --parallel --resume
  ```

See `python3 -m megdiffusion.pipeline.ddpm.train --help` for more information.
For other options like `batch_size`, we recommend modifying and backing up them in the yaml file.

If you want to sample with model trained by yourself (not the pre-trained model):

```shell
python3 -m megdiffusion.pipeline.ddpm.sample --nopretrain \
   --logdir ./path/to/logdir \
   --config ./configs/ddpm/cifar10.yaml  # Coule be your customed file
```

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
- [pesser](https://github.com/pesser)/[pytorch_diffusion](https://github.com/pesser/pytorch_diffusion): Unofficial PyTorch implementation of DDPM and provides converted torch checkpoints.
- [openai](https://github.com/openai)/[improved-diffusion](https://github.com/openai/improved-diffusion): The official codebase for Improved Denoising Diffusion Probabilistic Models.

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

```
@article{DBLP,
  title     = {Improved Denoising Diffusion Probabilistic Models},
  author    = {Alex Nichol and Prafulla Dhariwal},
  year      = {2021},
  url       = {https://arxiv.org/abs/2102.09672},
  eprinttype = {arXiv},
  eprint    = {2102.09672},
}
```