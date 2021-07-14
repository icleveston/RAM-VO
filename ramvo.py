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
        self.context = ContextNetwork(batch_size, device)
        self.core = CoreNetwork(batch_size, device)
        self.locator = LocationNetwork()
        self.regressor = RegressorNetwork() 
        self.baseliner = BaselineNetwork()

    def forward(self, x, op):
                
        l_t_array = []
        log_pi_array = []
        entropy_pi_array = []
        baseline_array = []
        
        # Generate the context for the first image
        h_t_0 = torch.zeros(self.batch_size, 1024).to(self.device)
        h_t_1 = self.context(op)
        
        # Generate the first location
        log_pi, l_t, entropy_pi = self.locator(h_t_1)
    
        # Generate the first baseline
        b_t = self.baseliner(h_t_1).squeeze()
        
        # Add the first observation to array
        log_pi_array.append(log_pi)
        l_t_array.append(l_t)
        baseline_array.append(b_t)
        entropy_pi_array.append(entropy_pi)
        
        # Initialize the latent space for each new mini batch
        self.core.hidden_cell = ((h_t_0.unsqueeze(0), torch.zeros(1, self.batch_size, 1024).to(self.device)),
                                 (h_t_1.unsqueeze(0), torch.zeros(1, self.batch_size, 1024).to(self.device)))

        for t in range(self.num_glimpses-1):
            
            # Capture the glimpses
            g_t = self.glimpse(x, l_t)
                
            # Integrate the input information from glimpses
            h_t_0, h_t_1 = self.core(g_t.unsqueeze(0))
            
            # Generate the next location
            log_pi, l_t, entropy_pi = self.locator(h_t_1)
            
            l_t_array.append(l_t)
            
            # Add the next observation to array
            log_pi_array.append(log_pi)
            entropy_pi_array.append(entropy_pi)
            
            # Generate the baseline
            b_t = self.baseliner(h_t_1).squeeze()
            
            # Add the baseline to array
            baseline_array.append(b_t)

        # Generate the prediction at the end
        predicted = self.regressor(h_t_0)
        
        # Stack the lists
        baseline_array = torch.stack(baseline_array)
        l_t_array = torch.stack(l_t_array)        
        log_pi_array = torch.stack(log_pi_array)
        entropy_pi_array = torch.stack(entropy_pi_array)
 
        return predicted, l_t_array, log_pi_array, baseline_array, entropy_pi_array


class ContextNetwork(nn.Module):

    def __init__(self, batch_size, device):
        super().__init__()
        
        self.batch_size = batch_size
        self.device = device
        
        self.conv_1_1 = nn.Conv2d(2, 16, kernel_size=5, padding=2)
        self.conv_1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2)
        self.conv_1_3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.conv_1_4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)

        self.fc_x = nn.Linear(6656, 1024)
            
    def forward(self, x):
                    
        # Reduce size
        k = x.shape[-1] // 192
        x = F.avg_pool2d(x, k)
        
        # Use CNN in each glimpse scale
        x = F.leaky_relu(self.conv_1_1(x), inplace=True)
        x = F.leaky_relu(self.conv_1_2(x), inplace=True)
        x = F.leaky_relu(self.conv_1_3(x), inplace=True)
        x = F.leaky_relu(self.conv_1_4(x), inplace=True)
        x = x.view(x.shape[0], -1)

        h = F.leaky_relu(self.fc_x(x), inplace=True)

        return h


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
        self.lstm_0 = nn.LSTM(512, 1024, 1)
        self.lstm_1 = nn.LSTM(1024, 1024, 1)
        
        self.hidden_cell = None

    def forward(self, g_t):
  
        _, state_0 = self.lstm_0(g_t, self.hidden_cell[0])
        
        h_t_0 = state_0[0]
        
        _, state_1 = self.lstm_1(h_t_0.detach(), self.hidden_cell[1])

        h_t_1 = state_1[0]

        self.hidden_cell = (state_0, state_1)
        
        return h_t_0[0], h_t_1[0]


class RegressorNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(1024, 256)
        
        self.fc_lr_1 = nn.Linear(256, 32)
        self.fc_lr_2 = nn.Linear(32, 3)
        
        self.fc_lt_1 = nn.Linear(256, 32)
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


class LocationNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc_mu_1 = nn.Linear(1024, 256)
        self.fc_mu_2 = nn.Linear(256, 32)
        self.fc_mu_3 = nn.Linear(32, 2)
        
        self.fc_std_1 = nn.Linear(1024, 256)
        self.fc_std_2 = nn.Linear(256, 32)
        self.fc_std_3 = nn.Linear(32, 2)
        
        torch.nn.init.orthogonal_(self.fc_mu_2.weight)
        torch.nn.init.orthogonal_(self.fc_mu_3.weight)
        torch.nn.init.orthogonal_(self.fc_std_2.weight)
        torch.nn.init.orthogonal_(self.fc_std_3.weight)
        
        #self.apply(weights_init_)
        
    def forward(self, h_t):
               
        # Compute mean
        mu = torch.tanh(self.fc_mu_1(h_t.detach()))
        mu = torch.tanh(self.fc_mu_2(mu))
        mu = torch.tanh(self.fc_mu_3(mu))
        
        # Compute the std
        log_std = torch.relu(self.fc_std_1(h_t.detach()))
        log_std = torch.relu(self.fc_std_2(log_std))
        log_std = self.fc_std_3(log_std)
                
        log_std = torch.clamp(log_std, min=-20, max=0)
        std = log_std.exp()
             
        # Create the normal dist
        normal = Normal(mu, std)
                
        x_t = normal.rsample()
        l_t = torch.tanh(x_t)
        
        log_pi = normal.log_prob(x_t)
        
        # Bound action
        log_pi -= torch.log(1 - l_t.pow(2) + 1e-6)
        log_pi = torch.sum(log_pi, dim=1)
        
        # Calc policy entropy
        entropy_pi = torch.sum(normal.entropy(), dim=1)
        
        return log_pi, l_t.detach(), entropy_pi.detach()


class BaselineNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc_1 = nn.Linear(1024, 256)
        self.fc_2 = nn.Linear(256, 32)
        self.fc_3 = nn.Linear(32, 1)

    def forward(self, h_t):
        
        b_t = torch.tanh(self.fc_1(h_t))
        b_t = torch.tanh(self.fc_2(b_t))
        b_t = self.fc_3(b_t)
        
        return b_t
    
    