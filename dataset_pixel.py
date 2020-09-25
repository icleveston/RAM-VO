import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

motion_options = [
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1)
]


class PixelDataset(Dataset):

    def __init__(self, trans):
           
        self.trans = trans
           
        data_dir = 'dataset/pixel/00/'
        
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
        return self.length

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Open image
        image_t = cv2.imread(os.path.join(self.path_frames, f"{idx}.jpg"))
        image_t_1 = cv2.imread(os.path.join(self.path_frames, f"{idx+1}.jpg"))
        
        #i = np.asarray(image_t)[:,:,0].ravel()
        #print(i.mean())
        #print(i.std())
        #exit()
        
        # Concat the two images
        images = [self.trans(image_t), self.trans(image_t_1)]
        #images = [image_t, image_t_1]
        
        # Read the ground truth
        coordinates_t = self.content[idx].split(',')
        coordinates_t_1 = self.content[idx+1].split(',')
        
        # Convert the coordinates to int
        x_t = int(coordinates_t[0])
        y_t = int(coordinates_t[1])
        x_t_1 = int(coordinates_t_1[0])
        y_t_1 = int(coordinates_t_1[1])
        
        # Compute the displacement
        #displacement = np.asarray([x_t_1-x_t, y_t_1-y_t])
        
        motion = (x_t_1-x_t, y_t_1-y_t)
        
        index = motion_options.index(motion)
            
        return images, index #displacement

