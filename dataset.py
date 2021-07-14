import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CityDataset(Dataset):

    def __init__(self, trans, preload=False):
           
        self.name = "City"
        self.trans = trans
        self.preload = preload

        data_dir = '/mnt/ssd/dataset/city/'
        
        self.path_frames = os.path.join(data_dir, 'images')
                
        self.length = 0
        self.gt = []
        self.images_array = []
            
        # Load the compressed file
        data = np.load(os.path.join(data_dir, "dataset.npz"))
        
        self.gt = data['y']
        self.images_array = data['x']
        self.length = len(self.gt)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_1 = Image.open(os.path.join(self.path_frames, f"{idx}.jpg")).convert('L')
        image_2 = Image.open(os.path.join(self.path_frames, f"{idx+1}.jpg")).convert('L')
            
        # Apply the transformations
        images = torch.stack([self.trans(image_1), self.trans(image_2)])
        
        # Read the ground truth
        coordinates_t = self.gt[idx].split(',')
        coordinates_t_1 = self.gt[idx+1].split(',')
        
        coordinates_t = list(map(float, coordinates_t))
        coordinates_t_1 = list(map(float, coordinates_t_1))
        
        # Convert the coordinates to int
        coordinates_t = torch.tensor(coordinates_t, dtype=torch.float)
        coordinates_t_1 = torch.tensor(coordinates_t_1, dtype=torch.float)
            
        return images, coordinates_t_1

    