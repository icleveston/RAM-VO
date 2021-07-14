import cv2
import os
import sys
sys.path.append(os.path.abspath("../../"))
from utils import noisy
        
path = '../../../dataset/kitti'

path_frames_array = [os.path.join(path, 'sequences', f"{i:02d}") for i in range(100, 111)]
path_gt_array = [os.path.join(path, 'poses' , f"{i:02d}.txt") for i in range(100, 111)]
        
for frames_path, gt_path in zip(path_frames_array, path_gt_array):
    
    content = None
    
    # Read the ground-truth lines
    with open(gt_path) as f:
        content = f.readlines()

    for idx in range(len(content)-1):

        img = cv2.imread(os.path.join(frames_path, 'image_0_processed_original', f"{idx:06d}.png"))
        
        noisy_img = noisy('s&p', img)

        cv2.imwrite(os.path.join(frames_path, 'image_0_processed', f"{idx:06d}.png"), noisy_img)
        
        print(idx)
ay):
    
    content = None
    
    # Read the ground-truth lines
    with open(gt_path) as f:
        content = f.readlines()

    for idx in range(len(content)-1):

        img = cv2.imread(os.path.join(frames_path, 'image_0_processed_original', f"{idx:06d}.png"))
        
        noisy_img = noisy('s&p', img)
        
        noisy_img = brightness(noisy_img, 0.5, 5)

        cv2.imwrite(os.path.join(frames_path, 'image_0_processed', f"{idx:06d}.png"), noisy_img)
        
