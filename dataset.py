import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
   
    
class PixelSkipped100Dataset(Dataset):

    def __init__(self, trans, trans_out, preload=False):
           
        self.name = "PixelSkipped100"
        self.trans = trans
        self.trans_out = trans_out
        self.preload = preload
           
        data_dir = '/mnt/ssd/dataset/pixel_skipped_100/'
        
        self.path_frames = os.path.join(data_dir, 'images')
        self.ground_truth = os.path.join(data_dir, "ground-truth.txt")
        
        self.content = []
        
        # Read the ground-truth lines
        with open(self.ground_truth) as f:
            self.content = f.readlines()
        
        # Remove the \n
        self.content = [x.strip() for x in self.content] 
        
        # Count the groundtruth lines
        self.length = len(self.content)-2
        
    def __len__(self):
        return self.length // 8

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image_1 = Image.open(os.path.join(self.path_frames, f"{idx}.jpg")).convert('L')
        image_2 = Image.open(os.path.join(self.path_frames, f"{idx+1}.jpg")).convert('L')
            
        # Apply the transformations
        images = torch.stack([self.trans(image_1), self.trans(image_2)])
        
        # Read the ground truth
        coordinates_t = self.content[idx].split(',')
        coordinates_t_1 = self.content[idx+1].split(',')
        
        # Convert the coordinates to int
        x_t = torch.tensor(float(coordinates_t[0]), dtype=torch.float)
        y_t = torch.tensor(float(coordinates_t[1]), dtype=torch.float)
        x_t_1 = torch.tensor(float(coordinates_t_1[0]), dtype=torch.float)
        y_t_1 = torch.tensor(float(coordinates_t_1[1]), dtype=torch.float)
        
        # Compute the displacement
        distance = self.trans_out(torch.stack([x_t_1-x_t, y_t_1-y_t]))
            
        return images, distance
    
    
    