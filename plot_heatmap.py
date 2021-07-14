import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
import torch
from utils import heatmap, annotate_heatmap, str2bool

image_size = 300

denormalize = lambda x, coords : (0.5 * ((coords + 1.0) * x)).long()


def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--dir", type=str, required=True, help="path to the output execution")
    arg.add_argument("--glimpse", type=int, required=True, help="glimpse to plot")
    arg.add_argument("--train", type=str2bool, required=False, help="should open the train data")
    
    args = vars(arg.parse_args())
    
    return args["dir"], args["glimpse"], args["train"]


def plot(glimpses_array, plot_dir, name):
    
    # Denormalize coordinates
    glimpses_array = denormalize(glimpses_array, image_size)

    x = glimpses_array[:, 0].cpu().data.numpy()
    y = glimpses_array[:, 1].cpu().data.numpy()

    heatmap_data, xedges, yedges = np.histogram2d(x, y, bins=75, range=[[-150, 150], [-150, 150]])
    
    fig, ax = plt.subplots(figsize=(10,10))
    fig.tight_layout()
    
    row_names = list(range(1, 51))
    col_names = list(range(1, 51))
    
    im, _ = heatmap(heatmap_data.T, row_names, col_names, ax=ax, cmap='magma_r', cbarlabel="total glimpses")
    
    #annotate_heatmap(im, valfmt="{x:.0f}", size=7, threshold=np.amax(heatmap_data)//3, textcolors=("red", "white"))
    
    plt.savefig(os.path.join("out", plot_dir, f'{name}'), orientation='landscape', dpi=100, bbox_inches='tight', pad_inches=0)
    
    plt.close(fig)


def main(plot_dir, glimpse, train):
    
    # Read the pickle files
    if train:
        
        # Get all the files inside the dir
        heatmaps_raw = [f.lower() for f in os.listdir(os.path.join("out", plot_dir))] 
        
        # Create the results folder
        if not os.path.exists(f'out/{plot_dir}/results'):
            os.makedirs(f'out/{plot_dir}/results')

        # For each image
        for filename in heatmaps_raw:
            
            glimpses_array = pickle.load(open(os.path.join("out", plot_dir, filename), "rb"))
            
            # Get the filename without extension
            name = filename.split('.')[0]
        
            if glimpse == 0:
                glimpses_array = glimpses_array.view(-1, 2)
                name = f'all_{name}.pdf'
            else:
                glimpses_array = glimpses_array[glimpse-1]
                name = f'{glimpse}_{name}.pdf'
        
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
    
    
