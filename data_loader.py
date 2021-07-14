import numpy as np
import torch
from torchvision import transforms
from torch.utils.data.sampler import *
from torch.utils.data import DataLoader
from dataset import KittiDatasetOriginal, EurocDataset
from utils import *
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_printoptions(profile="full", precision=10, linewidth=100, sci_mode=False)

class SequenceSampler(Sampler):
    
    def __init__(self, indices, batch_size):
          
        indices = np.asarray(indices)
        
        max_index = (len(indices)//batch_size)*batch_size
        
        self.final_indices = indices[:max_index].reshape(batch_size, -1).T.flatten()
        
    def __len__(self):
        return len(self.final_indices)
        
    def __iter__(self):
                
        return iter(self.final_indices)


def get_data_loader(batch_size, dataset, train_seq, val_seq, test_seq,
                    num_workers=4, pin_memory=False, preload=False, seed=1):

    # Set the seed
    torch.manual_seed(seed)
    
    train_loader = None
    valid_loader = None
    test_loader = None

    # Compose the transformations
    trans_input = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4209265411], [0.2889825404])
    ])

    mean = torch.tensor([-7.6397992e-05, 2.6872402e-04, 4.7161593e-06, -9.7197731e-04, -1.7675826e-02, 9.2309231e-01])
    std = torch.tensor([0.00305257, 0.01770405, 0.00267268, 0.02503707, 0.01716818, 0.30884704])

    trans_output = lambda x: (x - mean) / std

    # Call the dataset
    if train_seq is not None:
        dataset_train = KittiDatasetOriginal(sequences=train_seq, trans_input=trans_input, trans_output=trans_output, preload=preload)
         
        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=RandomSampler(dataset_train),
            #sampler=SequenceSampler(list(range(len(dataset_train))), batch_size),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
    
    if val_seq is not None:
        dataset_val = KittiDatasetOriginal(sequences=val_seq, trans_input=trans_input, trans_output=trans_output, preload=preload)
        
        valid_loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            sampler=RandomSampler(dataset_val),
            #sampler=SequenceSampler(list(range(len(dataset_val))), batch_size),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
    
    if test_seq is not None:
        dataset_test = KittiDatasetOriginal(sequences=[test_seq], trans_input=trans_input, trans_output=trans_output, preload=preload, should_skip=False)
    
        test_loader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset_test),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

    return train_loader, valid_loader, test_loader


def normalize_input(loader):

    mean, std = online_mean_and_sd(loader)
    
    print(mean, std)


def plot_dataset(loader):

    import cv2
    from utils import NormalizeInverse
    
    trans = transforms.Compose([
            #NormalizeInverse([0.5740], [0.3346]),
            transforms.ToPILImage()
    ])
    
    for i, (image, label) in enumerate(loader):
        
        first_image = image[0][0]
        second_image = image[0][1]
  
        first_image = trans(first_image)
        second_image = trans(second_image)
        
        first_image = cv2.cvtColor(np.array(first_image), cv2.COLOR_RGB2BGR)
        second_image = cv2.cvtColor(np.array(second_image), cv2.COLOR_RGB2BGR)
        
        # Show the image
        cv2.imshow("First", first_image)
        cv2.imshow("Second", second_image)
        
        print(label)

        cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows() 


def test_dataset():
    
    import cv2
    from utils import NormalizeInverse
    
    # Compose the transformations
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4561], [0.3082])
    ])
    
    transinv = transforms.Compose([
            NormalizeInverse([0.4561], [0.3082]),
            transforms.ToPILImage()
    ])

    # Call the dataset
    dataset_train = EurocDataset(sequences=[0, 1, 2, 5, 7, 8, 9], trans=trans, preload=True)
    
    image, coordinates = dataset_train[4538]
    
    first_image = image[0]
    second_image = image[1]

    first_image = transinv(first_image)
    second_image = transinv(second_image)

    first_image = cv2.cvtColor(np.array(first_image), cv2.COLOR_RGB2BGR)
    second_image = cv2.cvtColor(np.array(second_image), cv2.COLOR_RGB2BGR)

    # Show the image
    cv2.imshow("First", first_image)
    cv2.imshow("Second", second_image)
    
    cv2.waitKey()


def compute_dataset_motion(loader):

    gt_array = []

    for _, gt in loader:
        
        gt_array.append(gt[0].data.cpu().numpy())

    x = np.asarray(gt_array)
    
    return np.abs(x).mean(axis=0), np.abs(x).std(axis=0)


def normalize_output(loader):
    
    gt_array = []
    
    for _, gt in loader:
        
        gt_array.append(gt[0].data.cpu().numpy())

    x = np.asarray(gt_array)
    
    print(x.mean(axis=0))
    print(x.std(axis=0))


if __name__ == "__main__":
    
    import time
    import matplotlib.pyplot as plt
    
    #test_dataset()
    #exit()
           
    train_loader, _, _ = get_data_loader(1, dataset='kitti', train_seq=[0, 2, 4, 5, 6, 8, 9], val_seq=None, test_seq=None, preload=True)
 
    normalize_input(train_loader)
    normalize_output(train_loader)
    exit()
    
       
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        
        _, _, test_loader = get_data_loader(1, dataset='kitti', train_seq=None, val_seq=None, test_seq=i, preload=True)

        mean, std = compute_dataset_motion(test_loader)
    
        print(f"Seq {i}: mean={mean}, std={std}")
    
    exit()
    
    plot_dataset(train_loader)
    exit()   

    # import matplotlib.pyplot as plt
    
    # fig, axes = plt.subplots(1, 1)
    # fig.set_size_inches(5, 5)
    # axes.set_aspect('equal', adjustable='box')
    
    # major_ticks = np.arange(-100, 100, 5)
    # minor_ticks = np.arange(-100, 100, 1)

    # axes.set_xticks(major_ticks)
    # axes.set_xticks(minor_ticks, minor=True)
    # axes.set_yticks(major_ticks)
    # axes.set_yticks(minor_ticks, minor=True)

    # # And a corresponding grid
    # axes.grid(which='both')

    # # Or if you want different settings for the grids:
    # axes.grid(which='minor', alpha=0.5)
    # axes.grid(which='major', alpha=0.8)
    
    # axes.set_xlim(-100, 100)
    # axes.set_ylim(-100, 100)
    
    # last_point = np.array([0, 0])

    # # Get the first batch
    # for i, (images, labels) in enumerate(valid_loader):
        
    #     if i > 200:
    #         break
        
    #     last_point = last_point+labels[2].cpu().numpy()
        
    #     axes.plot(last_point[0], last_point[1], 'ro-')

    # plt.show()

    # Plot the batch
    plot_images(images, labels)
