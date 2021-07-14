import os
from PIL import Image
import numpy as np


def main():
    
    path = "../../../dataset/city"

    path_frames = os.path.join(path, "images")
    ground_truth = os.path.join(path, "ground-truth.txt")
    save_path = os.path.join(path, "dataset.npz")

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
        
        image = Image.open(os.path.join(path_frames, f"{i}.jpg")).convert('L')
        
        x.append(np.asarray(image))

    # Save the file
    with open(save_path, 'wb') as f:
        np.savez(f, x=x, y=y)


if __name__ == "__main__":

    main()
    
    
