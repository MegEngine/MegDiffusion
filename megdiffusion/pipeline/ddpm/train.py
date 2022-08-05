import os
import copy

import tqdm
import yaml
from absl import app, flags
from tensorboardX import SummaryWriter

import megengine as mge
import megengine.functional as F
import megengine.distributed as dist
import megengine.optimizer as optim
import megengine.autodiff as autodiff
from megengine import Tensor

from ...data import build_dataloader
from ...optimizer import build_optimizer
from ...diffusion import GaussionDiffusion
from ...diffusion.schedule import build_beta_schedule
from ...model.ddpm import UNet
from ...model.ema import ema
from ...utils.transform import linear_scale, linear_scale_rev
from ...utils.vision import make_grid, save_image

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "./configs/ddpm/cifar10.yaml", help="configuration file")
flags.DEFINE_string("dataset_dir", "/data/datasets/CIFAR10", help="dataset path")
flags.DEFINE_string("logdir", "./logs/DDPM_CIFAR10_EPS", help="log directory")
flags.DEFINE_bool("resume", False, help="resume training from saved checkpoint")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_bool("dtr", False, help="enable MegEngine DTR algorithm")

def train():

    with open(FLAGS.config, "r") as file:
        config = yaml.safe_load(file)

    if FLAGS.parallel:
        num_worker = dist.get_world_size()
        rank = dist.get_rank()
    else:
        num_worker = 1

    if FLAGS.dtr:
        mge.dtr.enable()
    
    # data setup
    train_dataloader = build_dataloader(
        dataset = config["data"]["dataset"],
        dataset_dir = FLAGS.dataset_dir,
        batch_size = config["training"]["batch_size"],
    )
    train_queue = iter(train_dataloader)

    # model setup
    model = UNet(**config["model"])
    ema_model = copy.deepcopy(model)

    optimizer = build_optimizer(
        params = model.parameters(),
        **config["optim"]["optimizer"],
    )
    gm = autodiff.GradManager()

    sample_path = os.path.join(FLAGS.logdir, "samples")
    checkpoint_path = os.path.join(FLAGS.logdir, "checkpoints")

    if FLAGS.resume:
        checkpoint = mge.load(os.path.join(checkpoint_path, "ckpt.pkl"))
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
    else:
        start_step = 0

    if num_worker > 1:
        dist.bcast_list_(model.tensors())
        gm.attach(model.parameters(), callbacks=[dist.make_allreduce_cb("sum")])
    else:
        gm.attach(model.parameters())
    
    # diffusion setup
    diffusion_config = config["diffusion"]
    diffusion = GaussionDiffusion(
        model=model,
        betas=build_beta_schedule(**diffusion_config["beta_schedule"]),
        model_mean_type=diffusion_config["model_mean_type"],
        model_var_type=diffusion_config["model_var_type"],
        loss_type=diffusion_config["loss_type"],
    )

    # logging pre-processing
    if num_worker == 1 or rank == 0:

        if not os.path.isdir(FLAGS.logdir):
            os.makedirs(FLAGS.logdir)
            os.makedirs(sample_path)
            os.makedirs(checkpoint_path)
        writer = SummaryWriter(FLAGS.logdir)

        # sample from real images for comparing

        real_batch_image, _ = next(iter(train_dataloader))
        real_grid_image = make_grid(real_batch_image)
        save_image(real_grid_image, os.path.join(sample_path, "real.png"))
        writer.add_image("real_image", real_grid_image)
        writer.flush()

    # train the model
    total_steps = config["training"]["n_iters"]
    sample_steps = config["training"]["n_sample"]
    validate_steps = config["training"]["n_validate"]
    save_steps = config["training"]["n_snapshot"]

    worker_steps = total_steps // num_worker
    start_step = start_step // num_worker
    
    with tqdm.trange(start_step, worker_steps, dynamic_ncols=True) as pbar:
        for worker_step in pbar:
            step = worker_step * num_worker
            image, _ = next(train_queue)
            image = Tensor(linear_scale(image))

            with gm:
                loss = diffusion.training_loss(image)
                gm.backward(loss)

            if config["optim"]["grad_clip"]:
                optim.clip_grad_norm(model.parameters(), config["optim"]["grad_clip"])

            optimizer.step().clear_grad()
            ema(model, ema_model, config["training"]["ema_decay"])

            if num_worker > 1:
                loss = dist.functional.all_reduce_sum(loss) / num_worker

            if num_worker == 1 or rank == 0:
                # add log information
                writer.add_scalar("loss", loss.mean().item(), step)
                pbar.set_postfix(loss="%.3f" % loss.mean().item())

                # sample from generated images for comparing
                # TODO: Support distributed sampling
                if sample_steps > 0 and step and step % sample_steps == 0:
                    model.eval()
                    generated_batch_image = diffusion.p_sample_loop((
                        config["sampling"]["batch_size"], config["data"]["img_resolution"],
                        config["data"]["image_size"], config["data"]["image_size"]
                    ))
                    generated_batch_image = F.clip(generated_batch_image, -1, 1).numpy()
                    generated_batch_image = linear_scale_rev(generated_batch_image)
                    generated_grid_image = make_grid(generated_batch_image)
                    path = os.path.join(sample_path, "%d.png" % step)
                    save_image(generated_grid_image, path)
                    writer.add_image("generated_image", generated_grid_image, step)
                    writer.flush()
                    model.train()

                # save checkpoints
                if save_steps > 0 and step and step % save_steps == 0:
                    ckpt = {
                        "model": model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step
                    }
                    
                    mge.save(ckpt, os.path.join(checkpoint_path, "ckpt.pkl"))

                # TODO: evaluate
                if validate_steps > 0 and step and step % validate_steps == 0:
                    pass

    if num_worker == 1 or rank == 0:
        writer.close()

def main(argv):
    if FLAGS.parallel:
        dist.launcher(train)()
    else:
        train()

if __name__ == "__main__":
    app.run(main)