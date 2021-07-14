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
    
    args = vars(arg.parse_args())
    
    return args["dir"]

  
def _klt(img_1, img_2):
    
    img_1 = np.asarray(img_1)
    img_2 = np.asarray(img_2)
            
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 1000,
                           qualityLevel = 0.01,
                           minDistance = 1,
                           useHarrisDetector=True)
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (50,50),
                      maxLevel = 5,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))
        
    p0 = cv2.goodFeaturesToTrack(img_1, mask=None, **feature_params)
    
    mask = np.zeros((360, 1220, 3), dtype=np.uint8)
    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]    
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        
        a,b = new.ravel()
        c,d = old.ravel()
        
        mask = cv2.arrowedLine(mask, (a,b),(c,d), (0, 255, 0), 2, tipLength=0.5)
    
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_GRAY2BGR)
    
    img = cv2.add(img_2, mask)
    
    return img

    
def main(plot_dir):
        
    # Read the pickle files
    glimpses = pickle.load(open(os.path.join("out", plot_dir, f"glimpses_epoch_full.p"), "rb"))

    img_all = glimpses[0].squeeze()
    locs_all = glimpses[1].transpose(0,1)
    
    # Get the parameters from the filename
    parameters = plot_dir.split("_")[0].split("x")

    # Get the parameters
    num_glimpses = 8 #int(parameters[0])
    patch_size = 32 #int(parameters[1])
    num_patches = 1 #int(parameters[3])
    glimpse_scale = 3 #int(parameters[2])
    
    img_0 = None
                       
    for j, (img, locs) in enumerate(zip(img_all, locs_all)):
        
        if img_0 is None:
            img_0 = img
            continue
        
        img_1 = img
        
        # Denormalize coordinates
        coordinates_x = [denormalize(np.shape(img)[1], l) for l in locs[:, 0]]
        coordinates_y = [denormalize(np.shape(img)[0], l) for l in locs[:, 1]]
        
        coordinates = np.stack((coordinates_x, coordinates_y), axis=1)
        
        # Round and convert to int
        coordinates = np.floor(coordinates)
        coordinates = coordinates.astype(int)

        # Create the plots
        fig, glimpse_0 = plt.subplots(nrows=1, ncols=1, figsize=(15,7))
        fig.tight_layout(rect=[0.005, 0, 1, 0.95], pad=0, w_pad=0, h_pad=0)
        fig.set_dpi(80)
        glimpse_0.xaxis.set_visible(False)
        glimpse_0.yaxis.set_visible(False)
            
        color = ["r", 'b', 'g', 'y']
        
        img = _klt(img_0.unsqueeze(0), img_1.unsqueeze(0))
                               
        # Set the image
        glimpse_0.imshow(img, cmap='gray')
        
        img_0 = img_1
        
        for i in range(num_glimpses):
            
            # Get the glimpse center
            glimpse_center = [coordinates[i][0], coordinates[i][1]]
            
            size = patch_size
            
            #Draw the glimpse for each num_patch
            for k in range(num_patches):

                #Draw the bounding box
                rect_0 = bounding_box(glimpse_center[0], glimpse_center[1], size, color[k])
                
                #Update the size
                size = size*glimpse_scale
                
                #Add to the subplot
                glimpse_0.add_patch(rect_0)
            
            
        plt.savefig(os.path.join("out", plot_dir, 'trajectory', f'glimpses_trajectory_{j}.png'), orientation='landscape', dpi=100, bbox_inches='tight', pad_inches=0)
    

if __name__ == "__main__":
    args = parse_arguments()
    
    main(args)
