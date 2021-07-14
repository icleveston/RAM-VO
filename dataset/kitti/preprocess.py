import cv2
import os

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))

load_path_frames = os.path.join('/mnt/ssd/dataset/kitti', 'sequences', '10', 'image_0')
save_path_frames = os.path.join('/mnt/ssd/dataset/kitti', 'sequences', '10', 'image_0_processed_large')
ground_truth = os.path.join('/mnt/ssd/dataset/kitti', 'poses' , '10.txt')
        
content = []
        
# Read the ground-truth lines
with open(ground_truth) as f:
    content = f.readlines()

# Remove the \n
content = [x.strip() for x in content] 

# Count the groundtruth lines
length = len(content)

x, y = (0, 0)
w, h = (1220, 360)

for idx in range(length):

    img = cv2.imread(os.path.join(load_path_frames, f"{idx:06d}.png"))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = img[y:y+h, x:x+w]
    
    # Equalize the histogram
    img = clahe.apply(img)
    
    img[img > 255] = 255
    img[img < 0] = 0

    cv2.imwrite(os.path.join(save_path_frames, f"{idx:06d}.png"), img) 
