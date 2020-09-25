import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from dataset_pixel import PixelDataset
from dataset_ball import BallDataset
from utils import plot_images


# Compose the transformations
trans = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.411, 0.411, 0.411), (10.191, 10.191, 10.191))
])


def get_train_valid_loader(
    batch_size,
    valid_size=0.2,
    num_workers=4,
    shuffle=True,
    pin_memory=False,
):
    """Train and validation data loaders.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        batch_size: how many samples per batch to load.
        valid_size: percentage split of the training set used for
            the validation set. Should be a float in the range [0, 1].
            In the paper, this number is set to 0.1.
        shuffle: whether to shuffle the train/validation indices.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """

    assert (valid_size >= 0) and (valid_size <= 1), "[!] valid_size should be in the range [0, 1]."

    # Call the dataset
    dataset = PixelDataset(trans=trans)
    #dataset = BallDataset(trans=trans)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader


def get_test_loader(data_dir, batch_size, num_workers=4, pin_memory=False):

    dataset = PixelDataset(trans=trans)
    #dataset = BallDataset(trans=trans)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader


if __name__ == "__main__":
       
    # Load 9 samples
    train_loader, _ = get_train_valid_loader(30)
 
    # Create the iterator
    train_loader = iter(train_loader)
    
    # Get the first batch
    images, labels = train_loader.next()

    # Plot the batch
    plot_images(images[0], labels)
