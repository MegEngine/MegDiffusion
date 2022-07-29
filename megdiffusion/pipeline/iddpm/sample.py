import os
import yaml
from absl import app, flags

import megengine as mge
import megengine.functional as F

from ...model.pretrain import iddpm_cifar10_uncond_50M_500K_converted
from ...model.iddpm import UNetModel
from ...diffusion import GaussionDiffusion
from ...diffusion.schedule import build_beta_schedule, cosine_schedule
from ...utils.transform import linear_scale_rev
from ...utils.vision import make_grid, save_image

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "./configs/iddpm/cifar10.yaml", help="configuration file")
flags.DEFINE_string("logdir", "./logs/IDDPM_CIFAR10_EPS", help="log directory")
flags.DEFINE_string("output_dir", "./output", help="output directory")
flags.DEFINE_boolean("ema", True, help="load ema model")
flags.DEFINE_boolean("pretrain", True, help="use pre-trained model")
flags.DEFINE_boolean("grid", True, help="make grid of the batch image")

def infer():

    with open(FLAGS.config, "r") as file:
        config = yaml.safe_load(file)

    # model setup
    if FLAGS.pretrain:
        model = iddpm_cifar10_uncond_50M_500K_converted(pretrained=True)
    else:  # use model trained from scratch
        assert os.path.isdir(FLAGS.logdir)
        checkpoint = mge.load(os.path.join(FLAGS.logdir, "checkpoints", "ckpt.pkl"))
        model = UNetModel(**config["model"])
        model.load_state_dict(checkpoint["model"])

    model.eval()

    # diffusion setup
    if FLAGS.pretrain:
        diffusion_config = model.diffusion_config
    else:
        diffusion_config = config["diffusion"]

    diffusion = GaussionDiffusion(
        model=model,
        betas=build_beta_schedule(**diffusion_config["beta_schedule"]),
        model_mean_type=diffusion_config["model_mean_type"],
        model_var_type=diffusion_config["model_var_type"],
        rescale_timesteps=diffusion_config["rescale_timesteps"],
    )

    generated_batch_image = diffusion.p_sample_loop((
        config["sampling"]["batch_size"], config["data"]["img_resolution"],
        config["data"]["image_size"], config["data"]["image_size"]
    ))
    generated_batch_image = F.clip(generated_batch_image, -1, 1).numpy()
    generated_batch_image = linear_scale_rev(generated_batch_image)


    if model.channel_order is not None:
        channel_order = model.channel_order
    else:
        channel_order = "BGR"
    
    if FLAGS.grid:
        generated_grid_image = make_grid(generated_batch_image)
        save_image(generated_grid_image, os.path.join(FLAGS.output_dir, "sample.png"), order=channel_order)
    else:  # save each image
        for idx, image in enumerate(generated_batch_image):
            save_image(image, os.path.join(FLAGS.output_dir, f"{idx}.png"), order=channel_order)

def main(argv):
    infer()

if __name__ == "__main__":
    app.run(main)