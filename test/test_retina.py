import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torchvision import transforms
from model import Retina
from utils import denormalize, bounding_box, str2bool
from data_loader import get_data_loader
torch.set_printoptions(threshold=10_000)

patch_size = 16
num_patches = 3
glimpse_scale = 2

# Define the glipmse position
loc = torch.Tensor([0.2872, 0.2272])
loc = loc.unsqueeze(0)

def main():

    path = "test/95.jpg"
    
    # Open the image
    img = cv2.imread(path)
    
    # Convert to tensor
    img = transforms.ToTensor()(img)
    
    img = img.unsqueeze(0) 

    _, _, test_loader = get_data_loader(
            512,
            0.1,
            0.05,
            4,
            False
        )
    
    g = iter(test_loader)
    x_batch, y_batch = next(g)
   
    img = x_batch[0][0]
    img = img.unsqueeze(0) 
    
    # Create the retina
    ret = Retina(patch_size=patch_size, num_patches=num_patches, glimpse_scale=glimpse_scale)
    
    # Get the glimpses
    glimpse = ret.foveate(img, loc)
    
    # Does it capture the point?
    catch = bool(torch.max(glimpse).item())
    
    glimpse = glimpse.data.numpy()
    
    # Convert to int
    glimpse = glimpse.astype(np.uint8)

    # Reshape the retina output
    glimpse = np.reshape(glimpse, [num_patches, 3, patch_size, patch_size])
    
    # Transpose the image cols
    glimpse = np.transpose(glimpse, [0, 2, 3, 1])
    img = np.transpose(img, [0, 2, 3, 1])

    # Create the subplot
    fig, axs = plt.subplots(nrows=1, ncols=num_patches+1)
    
    # Add the original image
    axs[0].imshow(img[0][:,:,0])
    
    coordinates = denormalize(np.shape(img)[2], loc)
    
    # Round and convert to int
    coordinates = np.floor(coordinates.data.numpy())
    glimpse_center = coordinates.astype(int)[0]
    
    # Set the plot title
    axs[0].set_title('c: '+ str(glimpse_center) + " - catch:" + str(catch), {'fontsize': 10, 'verticalalignment': 'top'}, y=1, loc='center')
    
    color = ["r", 'b', 'g', 'y']
    
    size = patch_size
    
    #Draw the glimpse for each num_patch
    for k in range(num_patches):

        #Draw the bounding box
        rect = bounding_box(glimpse_center[0], glimpse_center[1], size, color[k])
        
        #Update the size
        size = size*glimpse_scale
        
        #Add to the subplot
        axs[0].add_patch(rect)
    
    # For each subplot
    for i, ax in enumerate(axs.flat[1:]):
        
        print(glimpse[i])
        
        # Print the max value for the glimpse patch
        print("Scale " + str(i) + " = "+ str(np.amax(glimpse[i])))
        
        # Plot the glimpse
        axs[i+1].imshow(glimpse[i][:,:,0])
        
    plt.show()


if __name__ == "__main__":
    main()
