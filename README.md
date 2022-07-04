# MegDiffusion

MegEngine implementation of Diffusion Models. More details are coming soon.

Current maintainer: [@MegChai](https://github.com/MegChai)

## Usage
### Infer with pre-trained models (checkpoints)

Please make sure you have trained the model or gotten pre-trained model.

```
python3 -m megdiffusion.scripts.inference
```

The pre-trained models will be uploaded here soon.

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

## Todo list (welcome to contribute ✨)

- Datasets
  - [x] CIFAR10 (built-in)
  - [ ] CelebA-HQ
  - [ ] LSUN
- Evaluation (FID and Inception Score)
- Multi-GPUs inference
- Documentation
  - [ ] Docstrings for public API
  - [ ] Sphinx documentation
- Reproduce more experiments
- More improvements...

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