import os
from absl import app, flags

import megengine as mge
import megengine.functional as F

from ..model import ddpm_cifar10, ddpm_cifar10_ema
from ..model.ddpm import UNet
from ..diffusion import GaussionDiffusion
from ..utils.transform import linear_scale_rev
from ..utils.vision import make_grid, save_image

FLAGS = flags.FLAGS
# input (checkpoint) and ouput
flags.DEFINE_string("logdir", "./logs/DDPM_CIFAR10_EPS", help="log directory")
flags.DEFINE_string("outputdir", "./output", help="log directory")
flags.DEFINE_boolean("ema", True, help="load ema model")
flags.DEFINE_boolean("pretrain", True, help="use pre-trained model")
flags.DEFINE_boolean("grid", True, help="make grid of the batch image")
flags.DEFINE_integer("img_channels", 3, help="num of channels of training example")
flags.DEFINE_integer("img_resolution", 32, help="image size of training example")
flags.DEFINE_integer("sample_size", 64, "sampling size of images")
# model architecture
flags.DEFINE_integer("timesteps", 1000, help="total diffusion steps")
flags.DEFINE_integer("base_channel", 128, help="base channel of UNet")
flags.DEFINE_multi_integer("chanel_multiplier", [1, 2, 2, 2], help="channel multiplier")
flags.DEFINE_multi_integer("attention_resolutions", [16], help="resolutions use attension block")
flags.DEFINE_integer("num_res_blocks", 2, help="number of resblock in each downblock")
flags.DEFINE_float("dropout", 0.1, help="dropout rate of resblock")

def infer():
    # model setup
    if FLAGS.pretrain:
        model = ddpm_cifar10_ema(pretrained=True) if FLAGS.ema else ddpm_cifar10(pretrained=True)
    else:  # use model trained from scratch
        assert os.path.isdir(FLAGS.logdir)
        checkpoint = mge.load(os.path.join(FLAGS.logdir, "checkpoints", "ckpt.pkl"))
        model = UNet(FLAGS.timesteps, FLAGS.img_resolution, FLAGS.img_channels, FLAGS.img_channels,
            FLAGS.base_channel, FLAGS.chanel_multiplier, FLAGS.attention_resolutions,
            FLAGS.num_res_blocks, FLAGS.dropout)
        model.load_state_dict(checkpoint["ema_model" if FLAGS.ema else "model"])

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

    if FLAGS.grid:
        generated_grid_image = make_grid(generated_batch_image)
        save_image(generated_grid_image, os.path.join(FLAGS.outputdir, "sample.png"))
    else:  # save each image
        for idx, image in enumerate(generated_batch_image):
            save_image(image, os.path.join(FLAGS.outputdir, f"{idx}.png"))

def main(argv):
    infer()

if __name__ == "__main__":
    app.run(main)