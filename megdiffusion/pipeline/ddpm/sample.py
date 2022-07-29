import os
import yaml
from absl import app, flags

import megengine as mge
import megengine.functional as F

from ...model import pretrain
from ...model.ddpm import UNet
from ...diffusion import GaussionDiffusion
from ...diffusion.schedule import build_beta_schedule
from ...utils.transform import linear_scale_rev
from ...utils.vision import make_grid, save_image

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "./configs/ddpm/cifar10.yaml", help="configuration file")
flags.DEFINE_string("logdir", "./logs/DDPM_CIFAR10_EPS", help="log directory")
flags.DEFINE_string("output_dir", "./output", help="output directory")
flags.DEFINE_boolean("ema", True, help="load ema model")
flags.DEFINE_boolean("pretrain", True, help="use pre-trained model")
flags.DEFINE_boolean("grid", True, help="make grid of the batch image")

def infer():

    with open(FLAGS.config, "r") as file:
        config = yaml.safe_load(file)

    # model setup
    if FLAGS.pretrain:
        dataset_name = config["data"]["dataset"].lower()
        pretrained_model_type = ("ema_" if FLAGS.ema else "") + "converted"
        pretrained_model_name = f"ddpm_{dataset_name}_{pretrained_model_type}"
        model = getattr(pretrain, pretrained_model_name)(pretrained=True)
    else:  # use model trained from scratch
        assert os.path.isdir(FLAGS.logdir)
        checkpoint = mge.load(os.path.join(FLAGS.logdir, "checkpoints", "ckpt.pkl"))
        model = UNet(**config["model"])
        model.load_state_dict(checkpoint["ema_model" if FLAGS.ema else "model"])

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
    )

    # sample
    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(os.path.join(FLAGS.output_dir))

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