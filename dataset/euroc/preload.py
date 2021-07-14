import os
import csv
import re
from PIL import Image
import numpy as np


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def read_file_list(filename):

    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)


def associate(first_list, second_list, offset=0.0, max_difference=0.02):

    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches


def generate_frames_dict(path):
    
    images = [f.lower() for f in os.listdir(path)] 

    frames_dict = dict()

    # For each image
    for filename in sorted(images, key=natural_keys):
        frames_dict[float(filename.split('.')[0])] = filename
        
    return frames_dict


def main():
    
    path = '../../../dataset/euroc'
    
    sequence_range = ['MH_01_easy',
             'MH_02_easy',
             'MH_03_medium',
             'MH_04_difficult',
             'MH_05_difficult',
             'V1_01_easy',
             'V1_02_medium',
             'V1_03_difficult',
             'V2_01_easy',
             'V2_02_medium',
             'V2_03_difficult'
             ]

    path_frames_array = [os.path.join(path, i, 'cam0','data_processed') for i in sequence_range]
    path_gt_array = [os.path.join(path, i , 'state_groundtruth_estimate0', 'data.csv') for i in sequence_range]
    save_path_gt_array = [os.path.join(path, i , 'state_groundtruth_estimate0', 'data_processed.csv') for i in sequence_range]
    save_path_frames = os.path.join(path, "dataset.npz")
    
    sequences = []
    
    # For each sequence
    for frames_path, gt_path, save_gt_path in zip(path_frames_array, path_gt_array, save_path_gt_array):
        
        x = []
        y = []
        processed_gt = []
        
        # Load the ground truth dict
        gt_dict = read_file_list(gt_path)
        
        # Generate the
        images_dict = generate_frames_dict(frames_path)
        
        # Find the associations
        matches = associate(images_dict, gt_dict)
        
        # For each association
        for frame_index, gt_index in matches:
            
            # Load the image
            image = Image.open(os.path.join(frames_path, images_dict[frame_index])).convert('L')
            
            processed_gt.append([gt_index] + list(map(float,gt_dict[gt_index])))

            # Select only the xyz and quaternions
            gt = list(map(float, gt_dict[gt_index][0:7]))
            
            # Convert to numpy and add it to array
            x.append(np.asarray(image))
            y.append(gt)
            
        # Add the sequence to the array
        sequences.append([x, y])
        
        # Save processed gt
        with open(save_gt_path, "w") as f:
            writer = csv.writer(f, delimiter =',')
            writer.writerows(processed_gt)

    # Save the file
    with open(save_path_frames, 'wb') as f:
        np.savez(f, sequence=sequences)


if __name__ == "__main__":

    main()
    

