import os
from absl import app, flags

import megengine.functional as F

from ..model import ddpm_cifar10
from ..diffusion import GaussionDiffusion
from ..utils.transform import linear_scale_rev
from ..utils.vision import make_grid, save_image

FLAGS = flags.FLAGS
flags.DEFINE_integer("timesteps", 1000, help="total diffusion steps")
flags.DEFINE_integer("img_channels", 3, help="num of channels of training example")
flags.DEFINE_integer("img_resolution", 32, help="image size of training example")
flags.DEFINE_integer("sample_size", 64, "sampling size of images")
flags.DEFINE_string("outputdir", "./output", help="log directory")

def infer():
    # model setup
    model = ddpm_cifar10(pretrained=True)
    model.eval()

    # diffusion setup
    diffusion = GaussionDiffusion(FLAGS.timesteps, model)

    # sample
    if not os.path.isdir(FLAGS.outputdir):
        os.makedirs(os.path.join(FLAGS.outputdir))

    generated_batch_image = diffusion.p_sample_loop((
        FLAGS.sample_size, FLAGS.img_channels,
        FLAGS.img_resolution, FLAGS.img_resolution
    ))
    generated_batch_image = F.clip(generated_batch_image, -1, 1).numpy()
    generated_batch_image = linear_scale_rev(generated_batch_image)
    generated_grid_image = make_grid(generated_batch_image)
    path = os.path.join(FLAGS.outputdir, "sample.png")
    save_image(generated_grid_image, path)

def main(argv):
    infer()

if __name__ == "__main__":
    app.run(main)