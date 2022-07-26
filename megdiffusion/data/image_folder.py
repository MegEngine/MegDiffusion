import megengine.data as data
import megengine.data.transform as T
from megengine.data import Infinite
from megengine.data.dataset import ImageFolder

def build_imagefolder_dataloader(dataset_dir, batch_size):

    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToMode(),
    ])

    train_dataset = ImageFolder(root=dataset_dir)
    train_sampler = Infinite(data.RandomSampler(train_dataset, batch_size, drop_last=True))
    train_dataloader = data.DataLoader(train_dataset, train_sampler, transform)
    return train_dataloader
