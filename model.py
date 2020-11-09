import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


class RecurrentAttention(nn.Module):
    """A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References:
      [1]: Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(
        self, patch_size, num_patches, glimpse_scale, num_channels, hidden_glimpse, hidden_loc, std, hidden_size, num_out,
    ):
        """Constructor.

        Args:
          patch_size: size of the square patches in the glimpses extracted by the retina.
          num_patches: number of patches to extract per glimpse.
          glimpse_scale: scaling factor that controls the size of successive patches.
          num_channels: number of channels in each image.
          hidden_glimpse: hidden layer size of the fc layer for `phi`.
          hidden_loc: hidden layer size of the fc layer for `l`.
          std: standard deviation of the Gaussian policy.
          hidden_size: hidden size of the rnn.
          num_out: number of classes in the dataset.
          num_glimpses: number of glimpses to take per image,
            i.e. number of BPTT steps.
        """
        super().__init__()

        self.std = std

        self.sensor_0 = GlimpseNetwork(hidden_glimpse, hidden_loc, patch_size, num_patches, glimpse_scale, num_channels)
        self.sensor_1 = GlimpseNetwork(hidden_glimpse, hidden_loc, patch_size, num_patches, glimpse_scale, num_channels)
        self.rnn = CoreNetwork(hidden_size, hidden_size)
        self.locator_0 = LocationNetwork(hidden_size, 2, std)
        self.locator_1 = LocationNetwork(hidden_size, 2, std)
        self.regressor = ActionNetwork(hidden_size, num_out) 
        self.baseliner = BaselineNetwork(hidden_size, 1)

    def forward(self, x_0, x_1, l_t_prev_0, l_t_prev_1, h_state_prev, c_state_prev, last=False):
        """Run RAM for one timestep on a minibatch of images.

        Args:
            x: a 4D Tensor of shape (B, H, W, C). The minibatch
                of images.
            l_t_prev: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the previous
                timestep `t-1`.
            h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the previous timestep `t-1`.
            last: a bool indicating whether this is the last timestep.
                If True, the action network returns an output probability
                vector over the classes and the baseline `b_t` for the
                current timestep `t`. Else, the core network returns the
                hidden state vector for the next timestep `t+1` and the
                location vector for the next timestep `t+1`.

        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
            mu: a 2D tensor of shape (B, 2). The mean that parametrizes
                the Gaussian policy.
            l_t: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the
                current timestep `t`.
            b_t: a vector of length (B,). The baseline for the
                current time step `t`.
            log_probas: a 2D tensor of shape (B, num_classes). The
                output log probability vector over the classes.
            log_pi: a vector of length (B,).
        """
        predicted = None
        
        g_t_0, phi_0 = self.sensor_0(x_0, l_t_prev_0)
        g_t_1, phi_1 = self.sensor_1(x_1, l_t_prev_1)
        
        g_t = torch.cat((g_t_0, g_t_1), dim=1)
    
        #h_state, c_state = self.rnn(g_t, h_state_prev, c_state_prev)
        h_state = self.rnn(g_t, h_state_prev, c_state_prev)

        log_pi_0, l_t_0 = self.locator_0(h_state)
        log_pi_1, l_t_1 = self.locator_1(h_state)
        
        b_t = self.baseliner(h_state).squeeze()

        if last:
            predicted = self.regressor(h_state)

        return h_state, l_t_0, l_t_1, b_t, predicted, log_pi_0, log_pi_1, phi_0, phi_1


class Retina:
    """A visual retina.

    Extracts a foveated glimpse `phi` around location `l`
    from an image `x`.

    Concretely, encodes the region around `l` at a
    high-resolution but uses a progressively lower
    resolution for pixels further from `l`, resulting
    in a compressed representation of the original
    image `x`.

    Args:
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l: a 2D Tensor of shape (B, 2). Contains normalized
            coordinates in the range [-1, 1].
        patch_size: size of the first square patch.
        num_patches: number of patches to extract in the glimpse.
        glimpse_scale: scaling factor that controls the size of
            successive patches.

    Returns:
        phi: a 5D tensor of shape (batch_size, num_patches, patch_size, patch_size, num_channels).
        The foveated glimpse of the image.
    """

    def __init__(self, patch_size, num_patches, glimpse_scale):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.glimpse_scale = glimpse_scale

    def foveate(self, x, l):
        """Extract `num_patches` square patches of size `patch_size`, centered
        at location `l`. The initial patch is a square of
        size `patch_size`, and each subsequent patch is a square
        whose side is `glimpse_scale` times the size of the previous
        patch.

        The `num_patches` patches are finally resized to (patch_size, patch_size) and
        concatenated into a tensor of shape (B, num_patches, patch_size, patch_size, C).
        """
        phi = []
        size = self.patch_size

        # extract num_patches patches of increasing size
        for i in range(self.num_patches):
            phi.append(self._extract_patch(x, l, size))
            size = int(self.glimpse_scale * size)

        # resize the patches to squares of size patch_size
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.patch_size
            phi[i] = F.max_pool2d(phi[i], k) # avg_pool2d

        # concatenate into a single tensor and flatten
        phi = torch.cat(phi, 1)
        phi = phi.view(phi.shape[0], -1)

        return phi

    def _extract_patch(self, x, l, size):
        """Extract a single patch for each image in `x`.

        Args:
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l: a 2D Tensor of shape (B, 2).
        size: a scalar defining the size of the extracted patch.

        Returns:
            patch: a 4D Tensor of shape (B, size, size, C)
        """
        B, C, H, W = x.shape

        start = self.denormalize(H, l)
        end = start + size

        # pad with zeros
        x = F.pad(x, (size//2, size//2, size//2, size//2))
        
        # loop through mini-batch and extract patches
        patch = []
        for i in range(B):

            p = x[i, :, start[i, 1]:end[i, 1], start[i, 0]:end[i, 0]]

            patch.append(p)
        
        return torch.stack(patch)

    def denormalize(self, T, coords):
        """Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0) * T)).long()

    def exceeds(self, from_x, to_x, from_y, to_y, T):
        """Check whether the extracted patch will exceed
        the boundaries of the image of size `T`.
        """
        if (from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T):
            return True
        return False


class GlimpseNetwork(nn.Module):
    """The glimpse network.

    Combines the "what" and the "where" into a glimpse
    feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args:
        h_g: hidden layer size of the fc layer for `phi`.
        h_l: hidden layer size of the fc layer for `l`.
        patch_size: size of the square patches in the glimpses extracted
        by the retina.
        num_patches: number of patches to extract per glimpse.
        glimpse_scale: scaling factor that controls the size of successive patches.
        num_channels: number of channels in each image.
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
            coordinates [x, y] for the previous timestep `t-1`.

    Returns:
        g_t: a 2D tensor of shape (B, hidden_size).
            The glimpse representation returned by
            the glimpse network for the current
            timestep `t`.
    """

    def __init__(self, hidden_glimpse, hidden_loc, patch_size, num_patches, glimpse_scale, num_channels):
        super().__init__()

        # Create the retina
        self.retina = Retina(patch_size, num_patches, glimpse_scale)

        # Glimpse layers
        D_in = num_patches * patch_size * patch_size * num_channels
        self.fc_phi_1 = nn.Linear(D_in, hidden_glimpse)
        self.fc_phi_2 = nn.Linear(hidden_glimpse, hidden_glimpse)
        self.fc_phi_3 = nn.Linear(hidden_glimpse, hidden_glimpse)

        # Location layers
        self.fc_l_1 = nn.Linear(2, hidden_loc)
        self.fc_l_2 = nn.Linear(hidden_loc, hidden_loc)
        self.fc_l_3 = nn.Linear(hidden_loc, hidden_loc)
        
        self.fc_bn = nn.BatchNorm1d(hidden_glimpse+hidden_loc)

    def forward(self, x, l_t_prev):
        
        # Generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)

        phi_out_1 = F.relu(self.fc_phi_1(phi))
        phi_out_2 = F.relu(self.fc_phi_2(phi_out_1))
        phi_out_3 = self.fc_phi_3(phi_out_2)
        
        # Flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
        
        l_out_1 = F.relu(self.fc_l_1(l_t_prev))
        l_out_2 = F.relu(self.fc_l_2(l_out_1))
        l_out_3 = self.fc_l_3(l_out_2)

        # Concat the location and glimpses
        g_t = torch.cat((phi_out_3, l_out_3), dim=1)
        
        # Apply batch_norm
        g_t = F.relu(self.fc_bn(g_t))

        return g_t, phi


class CoreNetwork(nn.Module):
    """The core network.

    An RNN that maintains an internal state by integrating
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args:
        input_size: input size of the rnn.
        hidden_size: hidden size of the rnn.
        g_t: a 2D tensor of shape (B, hidden_size). The glimpse
            representation returned by the glimpse network for the
            current timestep `t`.
        h_t_prev: a 2D tensor of shape (B, hidden_size). The
            hidden state vector for the previous timestep `t-1`.

    Returns:
        h_t: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the current timestep `t`.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.i2h_1 = nn.Linear(input_size, hidden_size)
        self.i2h_2 = nn.Linear(hidden_size, hidden_size//2)
        
        self.h2h_1 = nn.Linear(hidden_size, hidden_size)
        self.h2h_2 = nn.Linear(hidden_size, hidden_size//2)
        
        self.fc_bn = nn.BatchNorm1d(hidden_size)
        
        #self.rnn = []
        
        #for i in range(1):
            
        #    if i == 0:
        #        self.rnn.append(nn.LSTMCell(input_size, hidden_size))
        #    else:
        #        self.rnn.append(nn.LSTMCell(hidden_size, hidden_size))

    def forward(self, g_t, h_state_prev, c_state_prev):
        
        h1_1 = F.relu(self.i2h_1(g_t))
        h1_2 = self.i2h_2(h1_1)
        
        h2_1 = F.relu(self.h2h_1(h_state_prev))
        h2_2 = self.h2h_2(h2_1)
        
        h_state = torch.cat((h1_2, h2_2), dim=1)
        
        h_state = F.relu(self.fc_bn(h_state))
        
        return h_state
        
        #h_state = []
        #c_state = []
                
        #for i in range(1):
            
        #    if i == 0:
        #        h_state_i, c_state_i = self.rnn[i](g_t, (h_state_prev[i], h_state_prev[i]))
        #    else:
        #        h_state_i, c_state_i = self.rnn[i](h_state_i, (h_state_prev[i], h_state_prev[i]))
            
        #    h_state.append(h_state_i)
        #    c_state.append(c_state_i)
        
        #return h_state, c_state


class ActionNetwork(nn.Module):
    """The action network.

    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        a_t: output probability vector over the classes.
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc_1_mu_x = nn.Linear(input_size, input_size//2)
        self.fc_2_mu_x = nn.Linear(input_size//2, input_size//4)
        self.fc_3_mu_x = nn.Linear(input_size//4, 1)
        
        self.fc_1_mu_y = nn.Linear(input_size, input_size//2)
        self.fc_2_mu_y = nn.Linear(input_size//2, input_size//4)
        self.fc_3_mu_y = nn.Linear(input_size//4, 1)

    def forward(self, h_t):
        
        mu_x_1 = F.relu(self.fc_1_mu_x(h_t))
        mu_x_2 = F.relu(self.fc_2_mu_x(mu_x_1))
        mu_x = torch.tanh(self.fc_3_mu_x(mu_x_2))
        
        mu_y_1 = F.relu(self.fc_1_mu_y(h_t))
        mu_y_2 = F.relu(self.fc_2_mu_y(mu_y_1))
        mu_y = torch.tanh(self.fc_3_mu_y(mu_y_2))
        
        # reparametrization trick
        #l_t_x = Normal(mu_x, 0.05).rsample()
        #l_t_y = Normal(mu_y, 0.05).rsample()

        # bound between [-1, 1]
        l_t_x = torch.clamp(mu_x, -1, 1)
        l_t_y = torch.clamp(mu_y, -1, 1)
        
        l_t = torch.cat((l_t_x, l_t_y), dim=1)

        return l_t


class LocationNetwork(nn.Module):
    """The location network.

    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        std: standard deviation of the normal distribution.
        h_t: the hidden state vector of the core network for
            the current time step `t`.

    Returns:
        mu: a 2D vector of shape (B, 2).
        l_t: a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, output_size, std):
        super().__init__()

        self.std = std

        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)

    def forward(self, h_t):
        
        # compute mean
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))
        
        # reparametrization trick
        l_t = Normal(mu, self.std).rsample()
        l_t = l_t.detach()
        log_pi = Normal(mu, self.std).log_prob(l_t)

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = torch.sum(log_pi, dim=1)

        # bound between [-1, 1]
        l_t = torch.clamp(l_t, -1, 1)

        return log_pi, l_t


class BaselineNetwork(nn.Module):
    """The baseline network.

    This network regresses the baseline in the
    reward function to reduce the variance of
    the gradient update.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        b_t: a 2D vector of shape (B, 1). The baseline
            for the current time step `t`.
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        return b_t
