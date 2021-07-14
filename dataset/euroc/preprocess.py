import cv2
import os
import re
        
path = '../../../dataset/euroc'

sequences = ['MH_01_easy',
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

x, y = (226, 90)
h, w = (300, 300)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

for seq in sequences:
    
    load_path_frames = os.path.join(path, seq, 'cam0', 'data')
    save_path_frames = os.path.join(path, seq, 'cam0','data_processed')
    
    images = [f.lower() for f in os.listdir(load_path_frames)] 

    # For each image
    for filename in sorted(images, key=natural_keys):

        img = cv2.imread(os.path.join(load_path_frames, filename))
        
        crop_img = img[y:y+h, x:x+w]

        cv2.imwrite(os.path.join(save_path_frames, filename), crop_img) 
