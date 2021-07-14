import cv2
import os
import numpy as np
from numpy import inf


load_path_frames = os.path.join('/mnt/ssd/dataset/kitti', 'sequences', '10', 'image_0_processed_large')
save_path_frames = os.path.join('/mnt/ssd/dataset/kitti', 'sequences', '10', 'optical_flow')
ground_truth = os.path.join('/mnt/ssd/dataset/kitti', 'poses' , '10.txt')
        
content = []
        
# Read the ground-truth lines
with open(ground_truth) as f:
    content = f.readlines()

# Remove the \n
content = [x.strip() for x in content] 

# Count the groundtruth lines
length = len(content)

prvs = None
hsv = None

for idx in range(length):
        
    if prvs is None:
        
        frame = cv2.imread(os.path.join(load_path_frames, f"{idx:06d}.png"))
        prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        
        continue
    
    # Read the next frame in
    frame = cv2.imread(os.path.join(load_path_frames, f"{idx:06d}.png"))
    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow between the two frames
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale = 0.5, levels = 10, winsize = 40, iterations = 10, poly_n = 7, poly_sigma = 1.5, flags = 0)
    
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    
    mag[mag > 255] = 0
    mag[mag < 0] = 0

    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Change - Make next frame previous frame
    prvs = next

    cv2.imwrite(os.path.join(save_path_frames, f"{(idx-1):06d}.png"), bgr)
    
    
