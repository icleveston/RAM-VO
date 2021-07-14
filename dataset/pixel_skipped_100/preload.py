import os
import cv2
import numpy as np
import argparse

def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", type=str, required=True, help="path to the output execution")
    
    args = vars(arg.parse_args())
    
    return args["data_dir"],

def main(data_dir):

    path_frames = os.path.join(data_dir, "00", "images")
    ground_truth = os.path.join(data_dir, "00", "ground-truth.txt")
    save_path = os.path.join(data_dir, "00", "dataset.npz")

    y = []

    # Read the ground-truth lines
    with open(ground_truth) as f:
        y = f.readlines()

    # Remove the \n
    y = [x.strip() for x in y] 

    # Count the groundtruth lines
    length = (len(y)-2)

    # The image array
    x = []

    for i in range(length):
        
        x.append(cv2.imread(os.path.join(path_frames, f"{i}.jpg")))

    # Save the file
    with open(save_path, 'wb') as f:
        np.savez(f, x=x, y=y)


if __name__ == "__main__":
    args = parse_arguments()

    main(*args)
    
    
