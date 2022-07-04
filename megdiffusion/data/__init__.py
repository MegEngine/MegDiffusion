from .cifar10 import build_cifar10_dataloader

__all__ = [
    "build_dataloader",
]

def build_dataloader(dataset, dataset_dir, batch_size):
    assert dataset in ["cifar10"]
    if dataset == "cifar10":
        return build_cifar10_dataloader(dataset_dir, batch_size)