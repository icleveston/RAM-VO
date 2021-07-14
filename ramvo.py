import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
torch.set_printoptions(profile="full", linewidth=200)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class RAMVO(nn.Module):

    def __init__(self, batch_size, patch_size, num_patches, num_glimpses, glimpse_scale, num_channels, device):

        super().__init__()

        self.batch_size = batch_size
        self.num_glimpses = num_glimpses
        self.device = device

        # Create the networks
        self.glimpse = GlimpseNetwork(patch_size, num_patches, glimpse_scale, num_channels)
        self.core = CoreNetwork(batch_size, device)
        self.regressor = RegressorNetwork() 

    def forward(self, x, l_t):
                
        # Capture the glimpses
        g_t = self.glimpse(x, l_t)
                
        # Integrate the input information from glimpses
        h_t_0, h_t_1 = self.core(g_t)

        # Generate the prediction at the end
        predicted = self.regressor(h_t_1)
                
        return h_t_0, h_t_1, predicted


class Retina:

    def __init__(self, patch_size, num_patches, glimpse_scale):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.glimpse_scale = glimpse_scale

    def foveate(self, x, l):

        phi = []
        size = self.patch_size

        # extract num_patches patches of increasing size
        for i in range(self.num_patches):
            phi.append(self._extract_patch(x, l, size))
            size = int(self.glimpse_scale * size)

        # resize the patches to squares of size patch_size
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.patch_size
            phi[i] = F.avg_pool2d(phi[i], k)

        # concatenate into a single tensor and flatten
        phi = torch.stack(phi)

        return phi

    def _extract_patch(self, x, l, size):

        B, C, H, W = x.shape

        # Denormalize dimension independently
        start_x = self.denormalize(W, l[:, 1])
        start_y = self.denormalize(H, l[:, 0])
               
        start = torch.stack((start_x, start_y), dim=1)
        
        end = start + size

        # pad with zeros
        x = F.pad(x, (size//2, size//2, size//2, size//2)) #, mode='reflect'
        
        # loop through mini-batch and extract patches
        patch = []
        for i in range(B):

            p = x[i, :, start[i, 1]:end[i, 1], start[i, 0]:end[i, 0]]

            patch.append(p)
        
        return torch.stack(patch)

    def denormalize(self, T, coords):

        return (0.5 * ((coords + 1.0) * T)).long()

    def exceeds(self, from_x, to_x, from_y, to_y, T):

        if (from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T):
            return True
        return False


class GlimpseNetwork(nn.Module):
   
    def __init__(self, patch_size, num_patches, glimpse_scale, num_channels):
        super().__init__()

        # Create the retina
        self.retina = Retina(patch_size, num_patches, glimpse_scale)

        # Image layers
        self.conv_1_1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv_1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv_1_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv_1_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_1_5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv_1_6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
        
        self.conv_2_1 = nn.Conv2d(2, 32, kernel_size=5, padding=2)
        self.conv_2_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=2)
        self.conv_2_3 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2)
        self.conv_2_4 = nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=2)
        
        self.conv_3_1 = nn.Conv2d(2, 32, kernel_size=5, padding=2)
        self.conv_3_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=2)
        self.conv_3_3 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2)
        self.conv_3_4 = nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=2)
        
        self.fc_x_1 = nn.Linear(2048, 256)
        self.fc_x_2 = nn.Linear(1024, 128)
        self.fc_x_3 = nn.Linear(1024, 128)
        
        self.fc_lt_1 = nn.Linear(2, 256)
        self.fc_lt_2 = nn.Linear(256, 512)

    def forward(self, x, l_t):
        
        # Generate glimpse piramid for both images
        x_0 = self.retina.foveate(x[:,0], l_t)
        x_1 = self.retina.foveate(x[:,1], l_t)
        
        # Separate the patches
        x_0_1, x_0_2, x_0_3 = x_0
        x_1_1, x_1_2, x_1_3 = x_1
        
        # Stack the patches from the two images
        x_1 = torch.cat((x_0_1, x_1_1), dim=1)
        x_2 = torch.cat((x_0_2, x_1_2), dim=1)
        x_3 = torch.cat((x_0_3, x_1_3), dim=1)
                
        x_1 = F.leaky_relu(self.conv_1_1(x_1), inplace=True)
        x_1 = F.leaky_relu(self.conv_1_2(x_1), inplace=True)
        x_1 = F.leaky_relu(self.conv_1_3(x_1), inplace=True)
        x_1 = F.leaky_relu(self.conv_1_4(x_1), inplace=True)
        x_1 = F.leaky_relu(self.conv_1_5(x_1), inplace=True)
        x_1 = F.leaky_relu(self.conv_1_6(x_1), inplace=True)
        x_1 = x_1.view(x_1.shape[0], -1)
        
        x_2 = F.leaky_relu(self.conv_2_1(x_2), inplace=True)
        x_2 = F.leaky_relu(self.conv_2_2(x_2), inplace=True)
        x_2 = F.leaky_relu(self.conv_2_3(x_2), inplace=True)
        x_2 = F.leaky_relu(self.conv_2_4(x_2), inplace=True)
        x_2 = x_2.view(x_2.shape[0], -1)
        
        x_3 = F.leaky_relu(self.conv_3_1(x_3), inplace=True)
        x_3 = F.leaky_relu(self.conv_3_2(x_3), inplace=True)
        x_3 = F.leaky_relu(self.conv_3_3(x_3), inplace=True)
        x_3 = F.leaky_relu(self.conv_3_4(x_3), inplace=True)
        x_3 = x_3.view(x_3.shape[0], -1)
        
        x_1 = self.fc_x_1(x_1)
        x_2 = self.fc_x_2(x_2)
        x_3 = self.fc_x_3(x_3)
                
        x_t = torch.cat([x_1, x_2, x_3], dim=1)
       
        l_t = l_t.view(l_t.size(0), -1)
        
        l_t = F.leaky_relu(self.fc_lt_1(l_t), inplace=True)    
        
        # What and where
        g_t = F.leaky_relu(x_t * self.fc_lt_2(l_t), inplace=True)
                        
        return g_t


class CoreNetwork(nn.Module):

    def __init__(self, batch_size, device):
        super().__init__()
        
        self.batch_size = batch_size
        self.device = device
        
        # LSTM with 2 layers
        self.lstm = nn.LSTM(512, 256, 2)
        
        self.hidden_cell = None

    def forward(self, g_t):
  
        _, self.hidden_cell = self.lstm(g_t.unsqueeze(0), self.hidden_cell)
 
        h_t_0 = self.hidden_cell[0][0]
        h_t_1 = self.hidden_cell[0][1]

        return h_t_0, h_t_1


class RegressorNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(256, 128)
        
        self.fc_lr_1 = nn.Linear(128, 32)
        self.fc_lr_2 = nn.Linear(32, 3)
        
        self.fc_lt_1 = nn.Linear(128, 32)
        self.fc_lt_2 = nn.Linear(32, 3)

    def forward(self, h_t):
        
        # General layer
        l_ = F.leaky_relu(self.fc(h_t), inplace=True)

        # Rotation layer
        l_r = F.leaky_relu(self.fc_lr_1(l_), inplace=True)
        l_r = self.fc_lr_2(l_r) # torch.tanh()
        
        # Translation layer
        l_t = F.leaky_relu(self.fc_lt_1(l_), inplace=True)
        l_t = self.fc_lt_2(l_t)
        
        pose = torch.cat((l_r, l_t), axis=1)

        return pose

