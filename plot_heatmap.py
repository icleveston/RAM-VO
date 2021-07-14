import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pickle
import argparse
import torch
from utils import heatmap, annotate_heatmap, str2bool

image_size_x = 1220
image_size_y = 360

denormalize = lambda x, coords : (0.5 * ((coords + 1.0) * x)).long()


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--dir", type=str, required=True, help="path to the output execution")
    arg.add_argument("--glimpse", type=int, required=True, help="glimpse to plot")
    arg.add_argument("--train", type=str2bool, required=False, help="should open the train data")
    
    args = vars(arg.parse_args())
    
    return args["dir"], args["glimpse"], args["train"]


def plot(glimpses_array, plot_dir, name):
    
    # Separate each coordinate
    glimpses_array_x = glimpses_array[:, 0]
    glimpses_array_y = glimpses_array[:, 1]
    
    # Denormalize coordinates
    glimpses_array_x = denormalize(glimpses_array_x, image_size_x)
    glimpses_array_y = denormalize(glimpses_array_y, image_size_y)
    
    x = glimpses_array_x.cpu().data.numpy()
    y = glimpses_array_y.cpu().data.numpy()

    heatmap_data, xedges, yedges = np.histogram2d(x, y, bins=(122,36), range=[[-610, 610], [-180, 180]])
    
    fig, ax = plt.subplots(figsize=(15,7))
    
    row_names = list(range(1, 37))
    col_names = list(range(1, 123))
    
    im, _ = heatmap(heatmap_data.T, row_names, col_names, ax=ax, cmap='magma_r', cbarlabel="total glimpses", interpolation='spline16')
    
    #annotate_heatmap(im, valfmt="{x:.0f}", size=6, threshold=np.amax(heatmap_data)//3, textcolors=("red", "white"))

    plt.savefig(os.path.join("out", plot_dir, f'{name}'), orientation='landscape', dpi=100, bbox_inches='tight', pad_inches=0)
    
    plt.close(fig)


def main(plot_dir, glimpse, train):
    
    # Read the pickle files
    if train:
        
        # Get all the files inside the dir
        heatmaps_raw = [f.lower() for f in os.listdir(os.path.join("out", plot_dir)) if os.path.isfile(os.path.join("out", plot_dir, f))] 
        
        # Create the results folder
        if not os.path.exists(f'out/{plot_dir}/results'):
            os.makedirs(f'out/{plot_dir}/results')

        # For each image
        for i, filename in enumerate(reversed(sorted(heatmaps_raw, key=natural_keys))):
            
            if i % 20 == 0:
            
                glimpses_array = pickle.load(open(os.path.join("out", plot_dir, filename), "rb"))
                
                # Get the filename without extension
                name = filename.split('.')[0]
            
                if glimpse == 0:
                    glimpses_array = glimpses_array[1:].view(-1, 2)
                    name = f'all_{name}.png'
                else:
                    glimpses_array = glimpses_array[glimpse-1]
                    name = f'{glimpse}_{name}.png'
            
                # Plot the heatmap
                plot(glimpses_array, f"{plot_dir}/results", name)


    
    else:
        
        glimpses_array = pickle.load(open(os.path.join("out", plot_dir, f"glimpses_heatmap.p"), "rb"))
    
        if glimpse == 0:
            glimpses_array = glimpses_array.view(-1, 2)
            name = 'heatmap_all.pdf'
        else:
            glimpses_array = glimpses_array[glimpse-1]
            name = f'heatmap_{glimpse}.pdf'
    
        # Plot the heatmap
        plot(glimpses_array, f"{plot_dir}", name)
    
    
if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
