import os
import copy
from sqlite3 import Time

import tqdm
from absl import app, flags
from tensorboardX import SummaryWriter

import megengine as mge
import megengine.functional as F
import megengine.distributed as dist
import megengine.optimizer as optim
import megengine.autodiff as autodiff
from megengine import Tensor

from ...data import build_dataloader
from ...model.ddpm import UNet
from ...diffusion import GaussionDiffusion
from ...utils.transform import linear_scale, linear_scale_rev
from ...model.ema import ema
from ...utils.vision import make_grid, save_image

FLAGS = flags.FLAGS
# dataset
flags.DEFINE_string("dataset", "cifar10", help="dataset used to train the model")
flags.DEFINE_integer("img_channels", 3, help="num of channels of training example")
flags.DEFINE_integer("img_resolution", 32, help="image size of training example")
flags.DEFINE_string("dataset_dir", "/data/datasets/CIFAR10", help="dataset path")
flags.DEFINE_integer("batch_size", 128, help="batch size for batch data from trainning dataset")
# model architecture
flags.DEFINE_integer("timesteps", 1000, help="total diffusion steps")
flags.DEFINE_integer("base_channel", 128, help="base channel of UNet")
flags.DEFINE_multi_float("channel_multiplier", [1, 2, 2, 2], help="channel multiplier")
flags.DEFINE_multi_integer("attention_resolutions", [16], help="resolutions use attension block")
flags.DEFINE_integer("num_res_blocks", 2, help="number of resblock in each downblock")
flags.DEFINE_float("dropout", 0.1, help="dropout rate of resblock")
# training
flags.DEFINE_bool("resume", False, help="resume training from saved checkpoint")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_bool("dtr", False, help="enable MegEngine DTR algorithm")
flags.DEFINE_float("lr", 0.0002, help="learning rate of the optimizer")
flags.DEFINE_float("grad_clip", 1., help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 800000, help="total training steps")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
# logging
flags.DEFINE_string("logdir", "./logs/DDPM_CIFAR10_EPS", help="log directory")
flags.DEFINE_integer("sample_size", 64, "sampling size of images")
flags.DEFINE_integer("sample_step", 1000, help="frequency of sampling, 0 to disable during training")
flags.DEFINE_integer("save_step", 5000, help="frequency of saving checkpoints")

def train():

    if FLAGS.parallel:
        num_worker = dist.get_world_size()
        rank = dist.get_rank()
    else:
        num_worker = 1

    if FLAGS.dtr:
        mge.dtr.enable()
    
    train_dataloader = build_dataloader(FLAGS.dataset, FLAGS.dataset_dir, FLAGS.batch_size)
    train_queue = iter(train_dataloader)

    # model setup
    model = UNet(
        total_timesteps = FLAGS.timesteps, 
        in_resolution = FLAGS.img_resolution, 
        in_channel = FLAGS.img_channels,
        out_channel = FLAGS.img_channels,
        base_channel = FLAGS.base_channel, 
        channel_multiplier = FLAGS.channel_multiplier, 
        attention_resolutions = FLAGS.attention_resolutions,
        num_res_blocks = FLAGS.num_res_blocks, 
        dropout = FLAGS.dropout,
    )
    ema_model = copy.deepcopy(model)

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
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
    diffusion = GaussionDiffusion(
        timesteps = FLAGS.timesteps, 
        model= model,
    )

    # logging pre-processing
    if num_worker == 1 or rank == 0:

        if not os.path.isdir(FLAGS.logdir):
            os.makedirs(FLAGS.logdir)
            os.makedirs(sample_path)
            os.makedirs(checkpoint_path)
        writer = SummaryWriter(FLAGS.logdir)

        # sample from real images for comparing
        real_batch_image = next(iter(train_dataloader))[0]
        real_grid_image = make_grid(real_batch_image[:FLAGS.sample_size])
        save_image(real_grid_image, os.path.join(sample_path, "real.png"))
        writer.add_image("real_image", real_grid_image)
        writer.flush()

        # backup all arguments
        with open(os.path.join(FLAGS.logdir, "flagfile.txt"), "w") as f:
            f.write(FLAGS.flags_into_string())

    # train the model
    start_step = start_step // num_worker
    worker_steps = FLAGS.total_steps // num_worker
    with tqdm.trange(start_step, worker_steps, dynamic_ncols=True) as pbar:
        for worker_step in pbar:
            step = worker_step * num_worker
            image, _ = next(train_queue)
            image = Tensor(linear_scale(image))

            with gm:
                loss = diffusion.p_loss(image)
                gm.backward(loss)

            # optim.clip_grad_norm(model.parameters(), FLAGS.grad_clip)
            optimizer.step().clear_grad()
            ema(model, ema_model, FLAGS.ema_decay)

            if num_worker > 1:
                loss = dist.functional.all_reduce_sum(loss) / num_worker

            if num_worker == 1 or rank == 0:
                # add log information
                writer.add_scalar("loss", loss.mean().item(), step)
                pbar.set_postfix(loss="%.3f" % loss.mean().item())

                # sample from generated images for comparing
                # TODO: Support distributed sampling
                if FLAGS.sample_step > 0 and step and step % FLAGS.sample_step == 0:
                    model.eval()
                    generated_batch_image = diffusion.p_sample_loop((
                        FLAGS.sample_size, FLAGS.img_channels,
                        FLAGS.img_resolution, FLAGS.img_resolution
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
                if FLAGS.save_step > 0 and step and step % FLAGS.save_step == 0:
                    ckpt = {
                        "model": model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step
                    }
                    
                    mge.save(ckpt, os.path.join(checkpoint_path, "ckpt.pkl"))

                # TODO: evaluate

    if num_worker == 1 or rank == 0:
        writer.close()

def main(argv):
    if FLAGS.parallel:
        dist.launcher(train)()
    else:
        train()

if __name__ == "__main__":
    app.run(main)