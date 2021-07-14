import os
import torch
import random
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from utils import *
    
 
class KittiDatasetOriginal(Dataset):

    def __init__(self, sequences, trans_input_images, trans_input_op, trans_output, preload=False, should_skip=True):
           
        # Set the seed
        torch.manual_seed(1)
        random.seed(1)
           
        self.name = "Kitti"
        self.trans_input_images = trans_input_images
        self.trans_input_op = trans_input_op
        self.trans_output = trans_output
        self.preload = preload

        data_dir = '/mnt/ssd/dataset/kitti/'
                
        self.sequence = []
        self.sequence_end = []
        
        # For each sequence to load
        for seq_id in sequences:
               
            # Load the compressed file
            data = np.load(os.path.join(data_dir, f"{seq_id:02d}.npz"), allow_pickle=False)
            total_frames = data['x']
            total_op = data['op']
            total_gt = data['y']
            
            # Create the blank
            blank = np.expand_dims(np.zeros_like(total_op[0]), axis=0)
            
            total_op = np.concatenate([blank, total_op])
            
            # Load the sequence
            self.sequence.append({'x':total_frames, 'op':total_op, 'y':total_gt})
            
            if len(self.sequence_end) == 0:
                self.sequence_end.append(len(total_gt))
            else:
                self.sequence_end.append(self.sequence_end[-1]+len(total_gt))
                        
            print(f"Loaded seq: {seq_id}")

    def __len__(self):
        return self.sequence_end[-1] - len(self.sequence_end)

    def __getitem__(self, idx):
        
        open_seq = None
        baseline = 0
        
        # Find the sequence to open
        for i, seq_end in enumerate(self.sequence_end):
            if idx < seq_end-1:
                open_seq = i
                break
            elif idx == seq_end-1:
                
                baseline = seq_end-1
                continue
            baseline = seq_end

        # Open the sequence
        sequence = self.sequence[open_seq]
        total_frames = sequence['x']
        total_op = sequence['op']
        total_gt = sequence['y']
        
        # Get the index
        index = idx - baseline             

        coordinate_prev = total_gt[index].split(' ')
        coordinate_prev = list(map(float, coordinate_prev))
        
        coordinate = total_gt[index+1].split(' ')
        coordinate = list(map(float, coordinate))

        #Compute the 6dof motion
        motion = get_ground_6d_poses(coordinate_prev, coordinate)
                        
        # Apply the transformations
        images = torch.stack([self.trans_input_images(total_frames[index]),
                              self.trans_input_images(total_frames[index+1]),
                            ])
            
        optical_flow = self.trans_input_op(total_op[index+1])
            
        # Apply the transformation to gt
        motion = self.trans_output(torch.tensor(motion, dtype=torch.float))
                       
        return images, optical_flow, motion
