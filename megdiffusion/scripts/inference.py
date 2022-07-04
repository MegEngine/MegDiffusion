import os
import tqdm
from absl import app, flags, logging

import megengine as mge
import megengine.functional as F

from ..model.ddpm import UNet
from ..diffusion import GaussionDiffusion
from ..utils.transform import linear_scale, linear_scale_rev
from ..utils.vision import make_grid, save_image

FLAGS = flags.FLAGS
# dataset
flags.DEFINE_integer("img_channels", 3, help="num of channels of training example")
flags.DEFINE_integer("img_resolution", 32, help="image size of training example")
# model
flags.DEFINE_integer("timesteps", 1000, help="total diffusion steps")
flags.DEFINE_integer("base_channel", 128, help="base channel of UNe")
flags.DEFINE_multi_integer("chanel_multiplier", [1, 2, 2, 2], help="channel multiplier")
flags.DEFINE_multi_integer("attention_resolutions", [16], help="resolutions use attension block")
flags.DEFINE_integer("num_res_blocks", 2, help="number of resblock in each downblock")
flags.DEFINE_float("dropout", 0.1, help="dropout rate of resblock")
# sample
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling, 0 to disable during training')

def infer():
    # model setup
    model = UNet(FLAGS.timesteps, FLAGS.img_resolution, FLAGS.img_channels, FLAGS.img_channels,
        FLAGS.base_channel, FLAGS.chanel_multiplier, FLAGS.attention_resolutions,
        FLAGS.num_res_blocks, FLAGS.dropout)

    # load checkpoint
    checkpoint = mge.load(os.path.join(FLAGS.logdir, "checkpoints", "ckpt.pkl"))
    model.load_state_dict(checkpoint["model"])

    # diffusion setup
    diffusion = GaussionDiffusion(FLAGS.timesteps, model)

    # sample
    if not os.path.isdir(FLAGS.logdir):
        os.makedirs(os.path.join(FLAGS.logdir))
    model.eval()
    generated_batch_image = diffusion.p_sample_loop((
        FLAGS.sample_size, FLAGS.img_channels,
        FLAGS.img_resolution, FLAGS.img_resolution
    ))
    generated_batch_image = F.clip(generated_batch_image, -1, 1).numpy()
    generated_batch_image = linear_scale_rev(generated_batch_image)
    generated_grid_image = make_grid(generated_batch_image)
    path = os.path.join(FLAGS.logdir, "sample.png")
    save_image(generated_grid_image, path)

def main(argv):
    infer()

if __name__ == "__main__":
    app.run(main)