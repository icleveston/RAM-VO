import cv2
import os
import re
import argparse

def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", type=str, required=True, help="path to the output execution")
    
    args = vars(arg.parse_args())
    
    return args["data_dir"],

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def main(data_dir):

    # Define the image path
    image_path = os.path.join('data', 'sequences', data_dir, 'image_0')    

    images = [f.lower() for f in os.listdir(image_path)] 

    # For each image
    for filename in sorted(images, key=natural_keys):
        
        # Read the image
        image = cv2.imread(os.path.join(image_path, filename))
        
        # Show the image
        cv2.imshow("Show Dataset", image)

        cv2.waitKey(100)

    # closing all open windows
    cv2.destroyAllWindows() 
    
    
if __name__ == "__main__":
    args = parse_arguments()

    main(*args)
