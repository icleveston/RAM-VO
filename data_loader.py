import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from dataset import PixelSkipped100Dataset
from utils import *


def normalize_input(loader):

    mean, std = online_mean_and_sd(loader)
    
    print(mean, std)
    
    
def normalize_output(loader):
    
    gt_array = []
    
    for _, gt in loader:
        
        gt_array.append(gt.view(-1).data.cpu().numpy())

    x = np.asarray(gt_array)
    
    print(x.mean())
    print(x.std())
    
    
def get_data_loader(
    batch_size,
    valid_size=0.2,
    test_size=0.1,
    num_workers=4,
    pin_memory=False,
    preload=False
):

    # Compose the transformations
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0004), (0.0200))
    ])
    
    trans_out = lambda x: (x - (-0.0037339155)) / 33.90598

    # Call the dataset
    dataset = PixelSkipped100Dataset(trans=trans, trans_out=trans_out, preload=preload)

    # Get the dataset size
    indices = list(range(len(dataset)))
    
    # Get the indexes for validation and test
    split_valid = int(np.floor(valid_size * len(dataset)))
    split_test = split_valid + int(np.floor(test_size * len(dataset)))

    # Split the dataset
    valid_idx, test_idx, train_idx = indices[:split_valid], indices[split_valid:split_test], indices[split_test:]

    # Define the subsampler for the dataset
    train_sampler = SequentialSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)
    test_sampler = SequentialSampler(test_idx)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
       
    # Load the samples
    train_loader, _, _ = get_data_loader(128)
    
    normalize_output(train_loader)
    normalize_input(train_loader)
    exit()
 
    # Create the iterator
    train_loader = iter(train_loader)
    
    # Get the first batch
    images, labels = train_loader.next()

    # Plot the batch
    plot_images(images, labels)
