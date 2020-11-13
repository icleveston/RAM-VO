import os
import pickle
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter 
from torchvision import transforms
from utils import denormalize, bounding_box, str2bool

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
    glimpses = pickle.load(open(os.path.join("out", plot_dir, 'glimpse', f"glimpses_epoch_{epoch}.p"), "rb"))
    
    img_0 = []
    img_1 = [] 
    locs_0 = []
    locs_1 = []
    catch_0 = []
    catch_1 = []
    
    for i in glimpses:
        img_0.append(i[0]) 
        img_1.append(i[1])
        locs_0.append(i[2]) 
        locs_1.append(i[3]) 
        catch_0.append(i[4]) 
        catch_1.append(i[5]) 
        
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
    patch_size = int(parameters[2])
    num_patches = int(parameters[3])
    glimpse_scale = int(parameters[4])
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
    fig.tight_layout(rect=[0.005, 0, 1, 0.95], pad=0, w_pad=0, h_pad=0)
    fig.set_dpi(80)

    def updateData(i):
        
        global image_index, input_index, first, interval_frames
        
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
            glimpse_0 = axs.flat[0]
            glimpse_1 = axs.flat[1]
                        
            # Set the image
            glimpse_0.imshow(img_0_denor[input_index][0,:,:])
            glimpse_1.imshow(img_1_denor[input_index][0,:,:])
            
            # Disable the axes
            #glimpse_0.get_yaxis().set_visible(False)
            #glimpse_0.get_xaxis().set_visible(False)
            #glimpse_1.get_yaxis().set_visible(False)
            #glimpse_1.get_xaxis().set_visible(False)
                    
            if i%num_glimpses == 0:
                image_index += 1
                    
            # Set the figure title
            fig.suptitle(f"Input: (t+{image_index}, t+{image_index+1}) | Glimpse step: {str((i%num_glimpses)+1)}", fontsize=14)
            
            # Get the glimpse center
            glimpse_center_0 = [coordinates_0[input_index][0], coordinates_0[input_index][1]]
            glimpse_center_1 = [coordinates_1[input_index][0], coordinates_1[input_index][1]]

            # Set the plot title
            glimpse_0.set_title('Sensor 0 - c: '+ str(glimpse_center_0) + " catch:" + str(catch_0[input_index]), {'fontsize': 10, 'verticalalignment': 'top'}, y=0, loc='center')
            glimpse_1.set_title('Sensor 1 - c: '+ str(glimpse_center_1) + " catch:" + str(catch_1[input_index]), {'fontsize': 10, 'verticalalignment': 'top'}, y=0, loc='center')
                
            # Remove all patches previously added
            glimpse_0.patches = []
            glimpse_1.patches = []
            
            size = patch_size
            
            #Draw the glimpse for each num_patch
            for k in range(num_patches):

                #Draw the bounding box
                rect_0 = bounding_box(glimpse_center_0[0], glimpse_center_0[1], size, color[k])
                rect_1 = bounding_box(glimpse_center_1[0], glimpse_center_1[1], size, color[k])
                
                #Update the size
                size = size*glimpse_scale
                
                #Add to the subplot
                glimpse_0.add_patch(rect_0)
                glimpse_1.add_patch(rect_1)
    
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
