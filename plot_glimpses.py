import pickle
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import transforms
from utils import denormalize, bounding_box

first = True
input_index = 1

def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--dir", type=str, required=True, help="path to directory")
    arg.add_argument("--epoch", type=int, required=True, help="epoch of desired plot")
    
    args = vars(arg.parse_args())
    
    return args["dir"], args["epoch"]


def main(plot_dir, epoch):
    
    # Read the pickle files
    glimpses = pickle.load(open(plot_dir + "glimpses_epoch_{}.p".format(epoch), "rb"))
    
    # Get the frame t
    img_0, locs_0 = glimpses[0]
    
    # Get the frame t+1
    img_1, locs_1 = glimpses[1]
    
    # Convert image to uint8
    img_0 = np.asarray(img_0).astype(np.uint8)
    img_1 = np.asarray(img_1).astype(np.uint8)
    
    img_0_denor = []
    img_1_denor = []
    
    # Denormalise to interval [0, 255]
    for i in img_0:
        img_0_denor.append(cv2.normalize(i, None, 0, 255, norm_type=cv2.NORM_MINMAX))
    for i in img_1:
        img_1_denor.append(cv2.normalize(i, None, 0, 255, norm_type=cv2.NORM_MINMAX))
    
    # Get the parameters from the filename
    parameters = plot_dir.split("_")

    # Get the parameters
    num_glimpses = int(parameters[1])
    patch_size = int(parameters[2][0])
    num_patches = int(parameters[-2][0])
    glimpse_scale = int(parameters[-1][0])
    img_shape = np.shape(img_0)[2]
    
    # Denormalize coordinates
    coordinates_0 = [denormalize(img_shape, l) for l in locs_0]
    coordinates_1 = [denormalize(img_shape, l) for l in locs_1]
    
    # Round and convert to int
    coordinates_0 = np.floor(coordinates_0)
    coordinates_1 = np.floor(coordinates_1)
    coordinates_0 = coordinates_0.astype(int)
    coordinates_1 = coordinates_1.astype(int)

    # Create the plots
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.set_dpi(300)

    def updateData(i):
        
        global input_index, first
        
        # First frame set white
        if first:
            first = False
            return
        
        color = ["r", 'b', 'g', 'y']
            
        # Get the subplot axes
        glimpse_0 = axs.flat[0]
        glimpse_1 = axs.flat[1]
        
        if i % num_glimpses == 0:
            
            # Update the input index
            input_index += 1

            # Set the plot title
            glimpse_0.set_title('Glimpse Sensor 0', {'fontsize': 10, 'verticalalignment': 'top'}, y=1, loc='center')
            glimpse_1.set_title('Glimpse Sensor 1', {'fontsize': 10, 'verticalalignment': 'top'}, y=1, loc='center')

            # Set the image
            glimpse_0.imshow(img_0_denor[input_index].T)
            glimpse_1.imshow(img_1_denor[input_index].T)
            
            # Disable the axes
            glimpse_0.get_yaxis().set_visible(False)
            glimpse_0.get_xaxis().set_visible(False)
            glimpse_1.get_yaxis().set_visible(False)
            glimpse_1.get_xaxis().set_visible(False)
                
        # Set the figure title
        fig.suptitle(f"Input: (t+{input_index}, t+{input_index+1}) | Glimpse step: {str((i%num_glimpses)+1)}", fontsize=14)
            
        # Remove all patches previously added
        glimpse_0.patches = []
        glimpse_1.patches = []
        
        size = patch_size
        
        #Draw the glimpse for each num_patch
        for k in range(1, num_patches+1):

            #Draw the bounding box
            rect_0 = bounding_box(coordinates_0[i%num_glimpses][input_index][0], coordinates_0[i%num_glimpses][input_index][1], size, color[k-1])
            rect_1 = bounding_box(coordinates_1[i%num_glimpses][input_index][0], coordinates_1[i%num_glimpses][input_index][1], size, color[k-1])
            
            #Update the size
            size = size*glimpse_scale
            
            #Add to the subplot
            glimpse_0.add_patch(rect_0)
            glimpse_1.add_patch(rect_1)
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, updateData, frames=num_glimpses*5, interval=500, repeat=True
    )

    # Save the video file
    name = plot_dir + "epoch_{}.mp4".format(epoch)
    anim.save(name, extra_args=["-vcodec", "h264", "-pix_fmt", "yuv420p"])


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
