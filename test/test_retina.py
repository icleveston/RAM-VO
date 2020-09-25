import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from model import Retina


def main():

    path = "test/4549.jpg" #"test/lenna.jpg"
    
    # Define the glipmse position
    loc = torch.Tensor([0.1, 0.1]) #[0.3, 0.05]
    loc = loc.unsqueeze(0)
    
    patch_size = 4 #25
    num_patches = 4 #5
    scale = 2

    # Open the image
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, dtype="float32")
    img = np.expand_dims(img, axis=0)
    
    # Convert to tensor
    img = torch.from_numpy(img)
    
    # Create the retina
    ret = Retina(patch_size=patch_size, num_patches=num_patches, glimpse_scale=scale)
    
    # Get the glimpses
    glimpse = ret.foveate(img, loc).data.numpy()
    
    # Convert to int
    glimpse = glimpse.astype(np.uint8)

    # Reshape the retina output
    glimpse = np.reshape(glimpse, [num_patches, 3, patch_size, patch_size])
    
    # Transpose the image cols
    glimpse = np.transpose(glimpse, [0, 2, 3, 1])
    
    # Reverse the stack
    glimpse = glimpse[::-1] 

    # Create the subplot
    fig, axs = plt.subplots(nrows=1, ncols=num_patches)
    
    # For each subplot
    for i, ax in enumerate(axs.flat):
        
        # Print the max value for the glimpse patch
        print(str(i) + " = "+ str(np.amax(glimpse[i])))
        
        # Plot the glimpse
        axs[i].imshow(glimpse[i])
        
        # Disable the axis
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
        
    plt.show()


if __name__ == "__main__":
    main()
