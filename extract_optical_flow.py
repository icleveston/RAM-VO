import cv2
import os
import re
import argparse
import numpy as np
from numpy import inf

def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--seq", type=str, required=True, help="the kitti sequence")
    arg.add_argument("--method", type=str, required=True, help="dense or sparse")
    
    args = vars(arg.parse_args())
    
    return args["seq"], args["method"]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def _farneback(seq):

    path = '/mnt/ssd/dataset/kitti/'

    # Define the image path
    image_path = os.path.join(path, 'sequences', f"{seq:02d}", 'image_0_processed_large')    

    images = [f.lower() for f in os.listdir(image_path)] 

    prvs = None
    hsv = None

    # For each image
    for filename in sorted(images, key=natural_keys):
        
        if prvs is None:
            
            frame = cv2.imread(os.path.join(image_path, filename))
            prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            
            continue
        
        # Read the next frame in
        frame = cv2.imread(os.path.join(image_path, filename))
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow between the two frames
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale = 0.5, levels = 10, winsize = 10, iterations = 10, poly_n = 7, poly_sigma = 1.5, flags = 0)
        
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        
        mag[mag == inf] = 0

        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        
        cv2.imshow('Original', next)
        cv2.imshow('Optical Flow', bgr)

        cv2.waitKey(100)
        
        # Change - Make next frame previous frame
        prvs = next

    cv2.destroyAllWindows()
    
    
def _klt(seq):
        
    path = '/mnt/ssd/dataset/kitti/'

    # Define the image path
    image_path = os.path.join(path, 'sequences', f"{seq:02d}", 'image_0_processed_large')    

    images = [f.lower() for f in os.listdir(image_path)] 

    old_gray = None
    p0 = None
    mask = None
    
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 1000,
                           qualityLevel = 0.01,
                           minDistance = 7,
                           useHarrisDetector=True)
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))

    # For each image
    for i, filename in enumerate(sorted(images, key=natural_keys)):
        
        if old_gray is None:
            
            # Take first frame and find corners in it
            frame = cv2.imread(os.path.join(image_path, filename))
            
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                            
            continue
                
        frame = cv2.imread(os.path.join(image_path, filename))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(frame)
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            
            a,b = new.ravel()
            c,d = old.ravel()
            
            mask = cv2.arrowedLine(mask, (a,b),(c,d), (0, 255, 0), 2,  tipLength=0.5)
           
        img = cv2.add(frame, mask)
        
        print(np.sum(mask))
        
        cv2.imshow('frame',img)
        
        cv2.waitKey(0)
    
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)        

        
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    
    args = parse_arguments()
    
    seq = int(args[0])
    
    if args[1] == 'sparse':
         _klt(seq)
    else:
        _farneback(seq)
    
    