import os
from PIL import Image
import numpy as np


def main():
    
    path = '/mnt/ssd/dataset/kitti'
    
    sequence_range = list(range(1, 11)) #+ list(range(100, 111))

    path_frames_array = [os.path.join(path, 'sequences', f"{i:02d}", 'image_0_processed_large') for i in sequence_range]
    path_op_array = [os.path.join(path, 'sequences',  f"{i:02d}", "optical_flow") for i in sequence_range]
    path_gt_array = [os.path.join(path, 'poses' , f"{i:02d}.txt") for i in sequence_range]
    path_save_array = [os.path.join(path, f"{i:02d}.npz") for i in sequence_range]
    
    for frames_path, op_path, gt_path, save_path in zip(path_frames_array, path_op_array, path_gt_array, path_save_array):
    
        y = []
        x = []
        op = []
    
        content = None
    
        # Read the ground-truth lines
        with open(gt_path) as f:
            content = f.readlines()
        
        # Remove the \n and concat
        y += [x.strip() for x in content]
        
        for i in range(len(content)):
            
            # Load the image
            image = Image.open(os.path.join(frames_path, f"{i:06d}.png")).convert('L')
                        
            # Convert to numpy and add it to array
            x.append(np.asarray(image))
            
            # Load the optical flow
            if i < len(content)-1:
                image_op = Image.open(os.path.join(op_path, f"{i:06d}.png"))
                op.append(np.asarray(image_op))           

        # Save the file
        with open(save_path, 'wb') as f:
            print(f"Images: {len(x)}")
            print(f"Groundtruth: {len(y)}")
            print(f"Optical Flow: {len(op)}")
            np.savez_compressed(f, x=x, op=op, y=y)


if __name__ == "__main__":

    main()
    

