import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from dataset_pixel import PixelUniformDataset, PixelSkipped100Dataset
from dataset_ball import BallDataset
from utils import plot_images


def get_data_loader(
    batch_size,
    valid_size=0.2,
    test_size=0.1,
    num_workers=4,
    pin_memory=False
):
    """Get the data loaders.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        batch_size: how many samples per batch to load.
        valid_size: percentage split of the training set used for
            the validation set. Should be a float in the range [0, 1].
            In the paper, this number is set to 0.1.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """

    assert (valid_size >= 0) and (valid_size <= 1), "[!] valid_size should be in the range [0, 1]."
    assert (test_size >= 0) and (test_size <= 1), "[!] test_size should be in the range [0, 1]."

    # Compose the transformations
    trans = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.411, 0.411, 0.411), (10.191, 10.191, 10.191))
    ])

    # Call the dataset
    #dataset = PixelUniformDataset(trans=trans)
    #dataset = PixelSkipped25Dataset(trans=trans)
    dataset = PixelSkipped100Dataset(trans=trans)
    #dataset = BallDataset(trans=trans)

    # Get the dataset size
    indices = list(range(len(dataset)))
    
    # Get the indexes for validation and test
    split_valid = int(np.floor(valid_size * len(dataset)))
    split_test = split_valid + int(np.floor(test_size * len(dataset)))

    # Split the dataset
    valid_idx, test_idx, train_idx = indices[:split_valid], indices[split_valid:split_test], indices[split_test:]

    # Define the subsampler for the dataset
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
       
    # Load 9 samples
    train_loader, _, _ = get_data_loader(3)
 
    # Create the iterator
    train_loader = iter(train_loader)
    
    # Get the first batch
    images, labels = train_loader.next()

    # Plot the batch
    plot_images(images, labels)
