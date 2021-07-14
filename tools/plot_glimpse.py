import sys
import os
import pickle
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter 
from torchvision import transforms
sys.path.append(os.getcwd())
from utils.utils import denormalize, bounding_box, str2bool

first = True
image_index = -1
input_index = -1
interval_frames = 15

def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--dir", type=str, required=True, help="path to the output execution")
    arg.add_argument("--epoch", type=str, required=True, help="epoch of desired plot")
    arg.add_argument("--save_as_gif", type=str2bool, default=True, help="save the plot as gif")
    
    args = vars(arg.parse_args())
    
    return args["dir"], args["epoch"], args["save_as_gif"]


def main(plot_dir, epoch, save_as_gif):
        
    # Read the pickle files
    glimpses = pickle.load(open(os.path.join("out", plot_dir, f"glimpses_epoch_{epoch}.p"), "rb"))

    img_0 = glimpses[0]
    img_1 = glimpses[1]
    locs = glimpses[2]
    
    # Get the parameters from the filename
    parameters = plot_dir.split("_")[0].split("x")

    # Get the parameters
    num_glimpses = 8 #int(parameters[0])
    patch_size = 32 #int(parameters[1])
    num_patches = 3 #int(parameters[3])
    glimpse_scale = 3 #int(parameters[2])
                       
    # Denormalize coordinates
    coordinates_x = [denormalize(np.shape(img_0)[1], l) for l in locs[:, 0]]
    coordinates_y = [denormalize(np.shape(img_0)[0], l) for l in locs[:, 1]]
    
    coordinates = np.stack((coordinates_x, coordinates_y), axis=1)
    
    # Round and convert to int
    coordinates = np.floor(coordinates)
    coordinates = coordinates.astype(int)

    # Create the plots
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15,7))
    fig.tight_layout(rect=[0.005, 0, 1, 0.95], pad=0, w_pad=0, h_pad=0)
    fig.set_dpi(80)

    def updateData(i):
        
        global image_index, input_index, first, interval_frames
        
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15,7))
        axs.xaxis.set_visible(False)
        axs.yaxis.set_visible(False)
        
        if i % interval_frames == 0:
            
            i = i//interval_frames
  
            # First frame set white
            if first:
                first = False
                return
            
            color = ["r", 'b', 'g', 'y']
            
            #Update the input index
            input_index += 1
                
            # Get the subplot axes
            glimpse_0 = axs
            #glimpse_1 = axs.flat[1]
                        
            # Set the image
            glimpse_0.imshow(img_0, cmap='gray')
            #glimpse_1.imshow(img_1, cmap='gray')
            
            # Disable the axes
            #glimpse_0.get_yaxis().set_visible(False)
            #glimpse_0.get_xaxis().set_visible(False)
            #glimpse_1.get_yaxis().set_visible(False)
            #glimpse_1.get_xaxis().set_visible(False)
                    
            if i%num_glimpses == 0:
                image_index += 1
                    
            # Set the figure title
            #fig.suptitle(f"Input: (t+{image_index}, t+{image_index+1}) | Glimpse step: {str((i%num_glimpses)+1)}", fontsize=14)
            
            # Get the glimpse center
            glimpse_center = [coordinates[input_index][0], coordinates[input_index][1]]

            # Set the plot title
            #glimpse_0.set_title('Sensor 0 - c: '+ str(glimpse_center), {'fontsize': 10, 'verticalalignment': 'top'}, y=1, pad=10, loc='center')
            #glimpse_1.set_title('Sensor 1 - c: '+ str(glimpse_center), {'fontsize': 10, 'verticalalignment': 'top'}, y=1, pad=10, loc='center')
                
            # Remove all patches previously added
            glimpse_0.patches = []
            #glimpse_1.patches = []
            
            size = patch_size
            
            #Draw the glimpse for each num_patch
            for k in range(num_patches):

                #Draw the bounding box
                rect_0 = bounding_box(glimpse_center[0], glimpse_center[1], size, color[k])
                #rect_1 = bounding_box(glimpse_center[0], glimpse_center[1], size, color[k])
                
                #Update the size
                size = size*glimpse_scale
                
                #Add to the subplot
                glimpse_0.add_patch(rect_0)
                #glimpse_1.add_patch(rect_1)
                
            plt.savefig(os.path.join("out", plot_dir, f'{i}.pdf'), orientation='landscape', dpi=100, bbox_inches='tight', pad_inches=0)
    
    # Create the animation
    anim = FuncAnimation(
        fig, updateData, frames=num_glimpses*interval_frames, repeat=False
    )

    # Save the video file
    if save_as_gif:
        anim.save(os.path.join("out", plot_dir, f"epoch_{epoch}.gif"), writer=ImageMagickWriter(fps=interval_frames))
    else:
        anim.save(os.path.join("out", plot_dir, f"epoch_{epoch}.mp4"), extra_args=["-vcodec", "h264", "-pix_fmt", "yuv420p"])


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
