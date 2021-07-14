import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from model import Retina
from utils import denormalize, bounding_box
from data_loader import get_data_loader
from utils import *
torch.set_printoptions(threshold=10_000)

patch_size = 48
num_patches = 3
glimpse_scale = 2

# Define the glipmse position
loc = torch.Tensor([0.2, 0.2])
loc = loc.unsqueeze(0)

color = ["r", 'b', 'g', 'y']


def main():
    
    # Create the retina
    ret = Retina(patch_size=patch_size, num_patches=num_patches, glimpse_scale=glimpse_scale)
    
    trans = transforms.Compose([
                    NormalizeInverse([0.2395], [0.1833]),
                    transforms.ToPILImage()
            ])

    train_loader, _, _ = get_data_loader(
            128
        )
    
    x_batch, y_batch = next(iter(train_loader))
   
    img_0 = x_batch[10][0].unsqueeze(0) 
    img_1 = x_batch[10][1].unsqueeze(0) 
    
    # Get the glimpses
    glimpse_0 = ret.foveate(img_0, loc)
    glimpse_1 = ret.foveate(img_1, loc)

    # Reshape the retina output
    glimpse_0 = glimpse_0.view(num_patches, 1, patch_size, patch_size)
    glimpse_1 = glimpse_1.view(num_patches, 1, patch_size, patch_size)
    
    coordinates = denormalize(np.shape(img_1)[2], loc)
    
    # Round and convert to int
    coordinates = np.floor(coordinates.data.numpy())
    glimpse_center = coordinates.astype(int)[0]

    # Create the subplot
    fig, axs = plt.subplots(nrows=2, ncols=4)
    
    # Add the original image
    axs[0][0].imshow(trans(img_0[0]), cmap='gray')
    axs[1][0].imshow(trans(img_1[0]), cmap='gray')

    # Set the plot title
    axs[0][0].set_title('c: '+ str(glimpse_center), {'fontsize': 10, 'verticalalignment': 'top'}, y=1, pad=10, loc='center')
    axs[1][0].set_title('c: '+ str(glimpse_center), {'fontsize': 10, 'verticalalignment': 'top'}, y=1, pad=10, loc='center')
    
    size = patch_size
    
    #Draw the glimpse for each num_patch
    for k in range(num_patches):

        #Draw the bounding box
        rect_0 = bounding_box(glimpse_center[0], glimpse_center[1], size, color[k])
        rect_1 = bounding_box(glimpse_center[0], glimpse_center[1], size, color[k])
        
        #Update the size
        size = size*glimpse_scale
        
        #Add to the subplot
        axs[0][0].add_patch(rect_0)
        axs[1][0].add_patch(rect_1)
    
    # For each subplot
    for i in range(3):
               
        # Plot the glimpse
        axs[0][i+1].imshow(trans(glimpse_0[i]), cmap='gray')
               
        # Plot the glimpse
        axs[1][i+1].imshow(trans(glimpse_1[i]), cmap='gray')

    plt.show()

if __name__ == "__main__":
    main()

