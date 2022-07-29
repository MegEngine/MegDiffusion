from .cifar10 import build_cifar10_dataloader
from .image_folder import build_imagefolder_dataloader

def build_dataloader(dataset: str, dataset_dir, batch_size):
    if dataset.lower() == "cifar10":
        return build_cifar10_dataloader(dataset_dir, batch_size)
    else:
        return build_imagefolder_dataloader(dataset_dir, batch_size)