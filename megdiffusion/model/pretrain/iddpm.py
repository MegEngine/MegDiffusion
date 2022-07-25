import megengine.hub as hub
from ..iddpm import UNetModel

@hub.pretrained("https://data.megengine.org.cn/research/megdiffusion/iddpm_cifar10_uncond_50M_500K_converted.pkl")
def iddpm_cifar10_uncond_50M_500K_converted(**kwargs):
    """The deault model configuration used in IDDPM paper on CIFAR10 dataset.
    Unconditional CIFAR-10 with our ``L_hybrid`` objective and cosine noise schedule
    Ported from: https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt

    Note:

        * MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
        * DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
        * TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

    """
    return UNetModel(
        in_channels=3,
        out_channels=6,
        model_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=3,
        attention_level=(2,4),
        dropout=0.3,
        use_scale_shift_norm=True,
        num_heads=4,
    )