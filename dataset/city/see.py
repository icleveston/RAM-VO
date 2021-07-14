import cv2
import os
import re
import argparse

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def main():

    images = [f.lower() for f in os.listdir(os.path.join('data', 'images'))] 

    # For each image
    for filename in sorted(images, key=natural_keys):
        
        # Read the image
        image = cv2.imread(os.path.join('data', 'images', filename))
        
        # Show the image
        cv2.imshow("Show Dataset", image)

        cv2.waitKey(100)

    # closing all open windows
    cv2.destroyAllWindows() 
    
    
if __name__ == "__main__":

    main()
