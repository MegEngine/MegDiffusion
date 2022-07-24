import argparse
import math
import os

import megengine as mge
import numpy as np
import torch
from megdiffusion.model.ddpm import UNet

from ckpt_utils import check_path, download, torch_cifar10_cfg, torch_lsun_cfg

"""
    MegEngine Model Structure:
        Unet {
            head {
                conv(w, b)
            },
            downblocks {
                resblock {
                    temb_proj,
                    block1 {
                        group norm(w, b),
                        swish,
                        conv(w, b)
                    },
                    block2 {
                        group norm(w, b),
                        swish,
                        drop_out,
                        conv(w, b)
                    },
                    temb_proj {
                        swish,
                        linear(w, b)
                    },
                    [
                        main {conv(w, b)},
                        short_cut {conv(w, b)}
                    ]
                }
            },
            middleblocks {
                resblock {
                    ...
                    [
                        attn {
                            group_norm,
                            proj,
                            proj_k,
                            proj_q,
                            proj_v
                        }
                    ...
                    ]
                }
            }
            upblocks {
                resblock {
                    ...
                }
            }
            tail {
                group norm(w, b),
                conv(w, b)
            }
        }
"""


def make_praser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-arch",
        "-m",
        default="cifar10",
        type=str,
        help="model architecture(optional: cifar10, ema_cifar10, lsun_bedroom, ema_lsun_bedroom, lsun_cat, \
            ema_lsun_cat, lsun_church, ema_lsun_church)",
    )
    parser.add_argument(
        "--torch-root",
        "-t",
        default="./torch_models/",
        type=str,
        help="path to download the pytorch pretrained model",
    )
    parser.add_argument(
        "--save-path",
        "-s",
        default="./checkpoints/",
        help="path to save the converted model",
    )
    parser.add_argument(
        "--verify",
        "-v",
        action='store_true',
        default=False,
        help="Whether to verify the converted model"
    )
    parser.add_argument(
        "--print",
        "-p",
        action='store_true',
        default=False,
        help="Whether to print convert infomation"
    )

    return parser


num_res_blocks = 2
block_size = 4

block_mapper = {
    'norm1': 'block1.0',
    'conv1': 'block1.2',
    'norm2': 'block2.0',
    'conv2': 'block2.3',
    'temb_proj': 'temb_proj.1',
    'nin_shortcut': 'short_cut'
}

attn_mapper = {
    'proj_out': 'proj',
    'q': 'proj_q',
    'k': 'proj_k',
    'v': 'proj_v',
    'norm': 'group_norm'
}

tail_mapper = {
    'norm_out': 'tail.0',
    'conv_out': 'tail.2'
}


def block_process(tar, ori):
    """
        Convert the torch key to megengine key in state dict.
        The torch key is formated as follow:
            down.0.block.0.norm1.weight
            ...
            down.0.block.1.norm1.bias
            ...
            down.0.downsample.conv.weight
            ...
            down.1.block.0.norm1.weight
        While megengine key like:
            downblocks.0.block1.0.weight
            ...
            downblocks.0.block1.2.bias
            ...
            downblocks.2.main.weight
            ...
            downblocks.3.block1.0.weight
        For upsample, the block idx is different:
            up.0.block.0.conv1.weight -> upblocks.12.block1.2.weight
            up.3.block.0.norm1.weight -> upblocks.0.block1.0.weight
        So we just need to replace block idx by block_size - 1 - block_idx, it will get the correct result.
    """
    upsample = tar == 'upblocks'
    # each downsample or upsample block in Unet has num_res_block ResBlock and one main block
    block_nums = num_res_blocks + 1 + upsample
    # bloack_size downsample and upsample blocks in Unet
    total_nums = block_nums * block_size - 1

    ori_name = ori.split('.')  # split origin name into list
    block_idx = int(ori_name[1])
    if upsample:
        block_idx = block_size - 1 - block_idx
    new_name = [tar]
    sub_block_name = ori_name[2]

    if sub_block_name == 'block':
        layer_idx = int(ori_name[3])
        idx = block_idx * block_nums + layer_idx
        new_name.extend([
            str(idx),
            block_mapper[ori_name[4]],
        ])
    elif sub_block_name == 'attn':
        new_name.extend([
            str(block_idx * block_nums + int(ori_name[3])),
            'attn',
            attn_mapper[ori_name[4]]
        ])
    else:  # for downsample and upsample
        new_name.extend([
            # down.0.downsample -> downblocks.(block_idx + 1) * block_nums - 1.main
            str((block_idx + 1) * block_nums - 1),
            'main',
        ])

    new_name.append(ori_name[-1])  # bias or weight

    return '.'.join(new_name)


def middle_process(tar, ori):
    ori_name = ori.split('.')
    sub_block_name, idx = ori_name[1].split('_')
    new_name = [tar, str(int(idx) - 1)]  # xxx_1 -> tar.0
    if sub_block_name == 'attn':
        new_name.extend([
            'attn',
            attn_mapper[ori_name[2]]
        ])
    else:
        new_name.append(block_mapper[ori_name[2]])

    new_name.append(ori_name[-1])  # bias or weight

    return '.'.join(new_name)


def replace_process(tar, ori):
    if tar == 'head':
        n = 'conv_in'
    elif tar == 'tail':
        n = ori.split('.')[0]
        tar = tail_mapper[n]
    else:
        n = ori.split('.')
        idx = n[2]
        # skip for Embedding in time_embedding which not exits in Pytorch implementation.
        if idx == '0':
            idx = '1'
        elif idx == '1':
            idx = '3'
        n = '.'.join(n[0:3])
        tar = '.'.join([tar, idx])
    return ori.replace(n, tar)


mapper = {
    'temb.dense': ("time_embedding.timembedding", replace_process),
    'conv_in': ('head', replace_process),
    'down': ('downblocks', block_process),
    'mid': ('middleblocks', middle_process),
    'up': ('upblocks', block_process),
    'conv_out': ('tail', replace_process),
    'norm_out': ('tail', replace_process),
}


def convert(old_dict, to_print):
    converted_dict = {}
    torch_key = list(old_dict.keys())

    for torch_name in torch_key:
        for origin_name, (target_name, process) in mapper.items():
            if origin_name in torch_name:
                # get the converted name by torch key
                new_name = process(target_name, torch_name)
                if to_print:
                    # blue color
                    print("convert key \033[1;34;40m{}\033[0m to \033[1;34;40m{}\033[0m".format(
                        torch_name, new_name))
                data = old_dict[torch_name].numpy()
                # reshape bias for convolution
                if 'timembedding' not in new_name and 'temb_proj' not in new_name and 'bias' in new_name:
                    data = data.reshape(1, -1, 1, 1)
                converted_dict[new_name] = data
                break
    return converted_dict


def _get_timestep_embedding(timesteps, embedding_dim):
    """Build sinusoidal embeddings, consider timesteps as num_embeddings."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype="float32") * -emb)
    pos = np.arange(timesteps, dtype="float32")  # discrete time step
    emb = pos[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)],
                         axis=1).reshape(timesteps, -1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = np.concatenate([emb, np.zeros(timesteps, 1)], axis=1)
    return emb


cifar10_cfg = {
    "total_timesteps": 1000,
    "in_resolution": 32,
    "in_channel": 3,
    "out_channel": 3,
    "base_channel": 128,
    "channel_multiplier": [1, 2, 2, 2],
    "attention_resolutions": [16],
    "num_res_blocks": 2,
    "dropout": 0.1,
}

lsun_cfg = {
    "total_timesteps": 1000,
    "in_resolution": 256,
    "in_channel": 3,
    "out_channel": 3,
    "base_channel": 128,
    "channel_multiplier": [1, 1, 2, 2, 4, 4],
    "attention_resolutions": [16],
    "num_res_blocks": 2,
    "dropout": 0.0,
}

model_config_map = {
    "cifar10": (cifar10_cfg, torch_cifar10_cfg),
    "lsun_bedroom": (lsun_cfg, torch_lsun_cfg),
    "lsun_cat": (lsun_cfg, torch_lsun_cfg),
    "lsun_church": (lsun_cfg, torch_lsun_cfg),
}


def main():
    global num_res_blocks, block_size

    args = make_praser().parse_args()
    assert args.model_arch in ['cifar10', 'ema_cifar10', 'lsun_bedroom', 'ema_lsun_bedroom', 'lsun_cat', 'ema_lsun_cat',
                               'lsun_church', 'ema_lsun_church'], "use python tools/convert.py --help to see the usage"

    check_path(args.save_path)
    check_path(args.torch_root)

    torch_model_path = download(args.torch_root, args.model_arch)
    torch_state_dict = torch.load(torch_model_path)

    config, torch_config = model_config_map[args.model_arch.replace(
        'ema_', '')]

    num_res_blocks = config['num_res_blocks']
    block_size = len(config['channel_multiplier'])

    state_dict = convert(torch_state_dict, args.print)

    # get timestep_embedding
    state_dict['time_embedding.timembedding.0.weight'] = _get_timestep_embedding(
        config['total_timesteps'], config['base_channel'])

    if args.verify:
        try:
            from pytorch_diffusion import Model
        except:
            raise ImportError(
                "Please install pytorch_diffusion from https://github.com/pesser/pytorch_diffusion, or close verify mode")
        mge_model = UNet(**config)
        mge_model.eval()
        mge_model.load_state_dict(state_dict)

        torch_model = Model(**torch_config)
        torch_model.eval()
        torch_model.load_state_dict(torch_state_dict)

        inp = np.random.randn(
            2, 3, config['in_resolution'], config['in_resolution'])
        t = np.array([5])
        torch_out = torch_model(torch.tensor(inp, dtype=torch.float32), torch.tensor(
            [5, ])).detach().numpy()
        mge_out = mge_model(mge.tensor(inp), mge.tensor(t)).numpy()
        print("allcolse verify result: ", np.allclose(
            mge_out, torch_out, atol=1e-6))
        print("The error is :", np.mean(abs(mge_out - torch_out)))

    save_path = os.path.join(args.save_path, args.model_arch + ".pkl")
    print("Saved converted model to ", save_path)
    mge.save(state_dict, save_path)


if __name__ == "__main__":
    main()
