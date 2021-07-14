import os
import torch
import random
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from utils import *
    
 
class KittiDatasetOriginal(Dataset):

    def __init__(self, sequences, trans_input, trans_output, preload=False, should_skip=True):
           
        # Set the seed
        torch.manual_seed(1)
        random.seed(1)
           
        self.name = "Kitti"
        self.trans_input = trans_input
        self.trans_output = trans_output
        self.preload = preload

        data_dir = '/mnt/ssd/dataset/kitti/'
                
        self.data = []
        
        # For each sequence to load
        for seq_id in sequences:
               
            # Load the compressed file
            data = np.load(os.path.join(data_dir, f"{seq_id:02d}.npz"), allow_pickle=False)
            total_frames = data['x']
            #total_op = data['op']
            total_gt = data['y']
            
            # Create the blank
            #blank = np.expand_dims(np.zeros_like(total_op[0]), axis=0)
            
            #total_op = np.concatenate([blank, total_op])
            
            #total_op = np.delete(total_op, 1, axis=3)
            
            # Load the sequence
            #self.sequence.append({'x':total_frames, 'op':total_op, 'y':total_gt})
            
            coordinate_prev = None
            frame_prev = None
         
            for frame, gt in zip(total_frames, total_gt):
                                 
                coordinate = gt.split(' ')
                coordinate = list(map(float, coordinate))
                                 
                if coordinate_prev is None:
                    coordinate_prev = coordinate
                    frame_prev = frame
                    continue
                        
                #Compute the 6dof motion
                gt_6dof = get_ground_6d_poses(coordinate_prev, coordinate)
                    
                self.data.append([frame_prev, frame, gt_6dof])
                
                # Reset 
                coordinate_prev = coordinate
                frame_prev = frame
            
            print(f"Loaded seq: {seq_id}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
                        
        image_1, image_2, motion = self.data[idx]
            
        # Apply the transformations
        images = torch.stack([self.trans_input(image_1), self.trans_input(image_2)])
                
        # Apply the transformation to gt
        motion = self.trans_output(torch.tensor(motion, dtype=torch.float))
            
        return images, motion
    
    
class EurocDataset(Dataset):

    def __init__(self, sequences, trans, preload=False):
           
        self.name = "Euroc"
        self.trans = trans
        self.preload = preload

        data_dir = '../dataset/euroc/'
                
        self.length = 0
        self.gt = []
        self.images_array = []
            
		# Load the compressed file
        data_sequence = np.load(os.path.join(data_dir, "dataset.npz"), allow_pickle = True)['sequence']
        
        self.seq_lengths = []
        
        for i in sequences:
        
            data = data_sequence[i]
            
            g = data[1][0:-1]

            self.gt += g
            self.images_array += data[0]
            self.length += len(g)
            
            if len(self.seq_lengths) == 0:
                self.seq_lengths.append(len(g))
            else:      
                self.seq_lengths.append(self.seq_lengths[-1]+len(g))
        

    def __len__(self):
        return self.length-12

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if (idx+1) in self.seq_lengths:
            idx += 1

        image_1 = self.images_array[idx]
        image_2 = self.images_array[idx+1]
            
        # Apply the transformations
        images = torch.stack([self.trans(image_1), self.trans(image_2)])
        
        # Read the ground truth
        coordinates_t = np.asarray(self.gt[idx])
        coordinates_t_1 = np.asarray(self.gt[idx+1])
        
        # Put w quaternion to the end
        coordinates_t_q = np.concatenate((coordinates_t[4:], np.asarray([coordinates_t[3]])), axis=0)
        coordinates_t_1_q = np.concatenate((coordinates_t_1[4:], np.asarray([coordinates_t_1[3]])), axis=0)

        coordinates_t_rotation_matrix = R.from_quat(coordinates_t_q).as_matrix()
        coordinates_t_1_rotation_matrix = R.from_quat(coordinates_t_1_q).as_matrix()
        
        # Join rotation and translation matrix
        coordinates_t_transf_matrix = np.concatenate((coordinates_t_rotation_matrix, coordinates_t[:3].reshape(-1, 1)), axis=1)
        coordinates_t_1_transf_matrix = np.concatenate((coordinates_t_1_rotation_matrix, coordinates_t_1[:3].reshape(-1, 1)), axis=1)
        
        # Compute the 6-dof motion
        motion = get_ground_6d_poses(coordinates_t_transf_matrix.flatten(), coordinates_t_1_transf_matrix.flatten())
        
        motion = torch.tensor(motion, dtype=torch.float)
            
        return images, motion
    
class PixelContinuous100Dataset(Dataset):

    def __init__(self, trans, preload=False):
           
        self.name = "PixelContinuous100"
        self.trans = trans
        self.preload = preload

        data_dir = 'dataset/pixel_continuous_100/00/'
        
        self.path_frames = os.path.join(data_dir, 'images')
        self.ground_truth = os.path.join(data_dir, "ground-truth.txt")
        
        self.content = []
        
        # Read the ground-truth lines
        with open(self.ground_truth) as f:
            self.content = f.readlines()
        
        # Remove the \n
        self.content = [x.strip() for x in self.content] 
        
        # Count the groundtruth lines
        self.length = (len(self.content)-2)//2
        
        # The image array
        self.images_array = []
        
        if self.preload:
            
            # Load the compressed file
            data = np.load(os.path.join(data_dir, "dataset.npz"))
            
            self.content = data['y']
            self.images_array = data['x']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.preload:
            image_1 = self.images_array[idx]
            image_2 = self.images_array[idx+1]
        else:
            image_1 = cv2.imread(os.path.join(self.path_frames, f"{idx}.jpg"))
            image_2 = cv2.imread(os.path.join(self.path_frames, f"{idx+1}.jpg"))
        
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
        distance = torch.stack([x_t_1-x_t, y_t_1-y_t])
            
        return images, distance
    
    
class PixelSkipped100Dataset(Dataset):

    def __init__(self, trans, preload=False):
           
        self.name = "PixelSkipped100"
        self.trans = trans
        self.preload = preload
           
        data_dir = 'dataset/pixel_skipped_100/00/'
        
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
        
        # The image array
        self.images_array = []
        
        if self.preload:
            
            # Load the compressed file
            data = np.load(os.path.join(data_dir, "dataset.npz"))
            
            self.content = data['y']
            self.images_array = data['x']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.preload:
            image_1 = self.images_array[idx]
            image_2 = self.images_array[idx+1]
        else:
            image_1 = cv2.imread(os.path.join(self.path_frames, f"{idx}.jpg"))
            image_2 = cv2.imread(os.path.join(self.path_frames, f"{idx+1}.jpg"))

        # Apply the transformations
        images = [self.trans(image_1), self.trans(image_2)]
        
        # Read the ground truth
        coordinates_t = self.content[idx].split(',')
        coordinates_t_1 = self.content[idx+1].split(',')
        
        # Convert the coordinates to int
        x_t = torch.tensor(float(coordinates_t[0]), dtype=torch.float)
        y_t = torch.tensor(float(coordinates_t[1]), dtype=torch.float)
        x_t_1 = torch.tensor(float(coordinates_t_1[0]), dtype=torch.float)
        y_t_1 = torch.tensor(float(coordinates_t_1[1]), dtype=torch.float)
        
        # Compute the displacement
        distance = torch.stack([x_t_1-x_t, y_t_1-y_t])
            
        return images, distance
    
