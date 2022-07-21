import os

from absl import app, flags

import megengine as mge
import megengine.functional as F

from ...model import iddpm_cifar10_uncond_50M_500K_converted
from ...model.iddpm import UNetModel
from ...diffusion import GaussionDiffusion
from ...utils.transform import linear_scale_rev
from ...utils.vision import make_grid, save_image
from ...diffusion.schedule import consine_schedule

FLAGS = flags.FLAGS
# input (checkpoint) and ouput
flags.DEFINE_string("logdir", "./logs/IDDPM_CIFAR10_UNCOND", help="log directory")
flags.DEFINE_string("output_dir", "./output", help="log directory")
flags.DEFINE_boolean("pretrain", True, help="use pre-trained model")
flags.DEFINE_boolean("grid", True, help="make grid of the batch image")
flags.DEFINE_integer("img_channels", 3, help="num of channels of training example")
flags.DEFINE_integer("img_size", 32, help="image size of training example")
flags.DEFINE_integer("sample_size", 64, "sampling size of images")
# model architecture
flags.DEFINE_integer("timesteps", 4000, help="total diffusion steps")
flags.DEFINE_integer("nums_channel", 128, help="base channel of UNet")
flags.DEFINE_multi_float("channel_multiplier", [1, 2, 2, 2], help="channel multiplier")
flags.DEFINE_multi_integer("attention_level", [2, 4], help="level use attension block")
flags.DEFINE_integer("num_res_blocks", 3, help="number of resblock in each downblock")
flags.DEFINE_float("dropout", 0.3, help="dropout rate of resblock")
flags.DEFINE_boolean("learn_sigma", True, help="learn the model variance range")
flags.DEFINE_boolean("use_scale_shift_norm", True, help="use scale shift norm")

def infer():
    # model setup
    if FLAGS.pretrain:
        model = iddpm_cifar10_uncond_50M_500K_converted(pretrained=True)
    else:  # use model trained from scratch
        assert os.path.isdir(FLAGS.logdir)
        checkpoint = mge.load(os.path.join(FLAGS.logdir, "checkpoints", "ckpt.pkl"))
        model = UNetModel(
            in_channels=FLAGS.img_channels,
            out_channels=FLAGS.img_channels * 2 if FLAGS.learn_sigma else FLAGS.img_channels,
            model_channels=FLAGS.nums_channel,
            channel_mult=FLAGS.nums_channel,
            num_res_blocks=FLAGS.num_res_blocks,
            attention_level=FLAGS.num_res_blocks,
            dropout=FLAGS.dropout,
            use_scale_shift_norm=FLAGS.use_scale_shift_norm,
        )
        model.load_state_dict(checkpoint["model"])

    model.eval()

    timesteps = FLAGS.timesteps

    betas = consine_schedule(timesteps)
    diffusion = GaussionDiffusion(
        timesteps = timesteps, 
        betas = betas,
        model = model,
        model_var_type="LEARNED_RANGE", 
        rescale_timesteps = True,
    )

    generated_batch_image = diffusion.p_sample_loop((
        FLAGS.sample_size, FLAGS.img_channels,
        FLAGS.img_size, FLAGS.img_size
    ))
    generated_batch_image = F.clip(generated_batch_image, -1, 1).numpy()
    generated_batch_image = linear_scale_rev(generated_batch_image)

    if FLAGS.grid:
        generated_grid_image = make_grid(generated_batch_image)
        save_image(generated_grid_image, os.path.join(FLAGS.output_dir, "sample.png"), "rgb")
    else:  # save each image
        for idx, image in enumerate(generated_batch_image):
            save_image(image, os.path.join(FLAGS.output_dir, f"{idx}.png"), "rgb")

def main(argv):
    infer()

if __name__ == "__main__":
    app.run(main)