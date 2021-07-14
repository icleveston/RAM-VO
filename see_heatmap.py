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
    image_path = os.path.join('out', data_dir, 'heatmap', 'results')    

    images = [f.lower() for f in os.listdir(image_path)]
    
    out = cv2.VideoWriter(f'{image_path}/output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, (3000, 3000))

    # For each image
    for filename in sorted(images, key=natural_keys):
        
        # Read the image
        image = cv2.imread(os.path.join(image_path, filename))
        
        # Show the image
        cv2.imshow(f"Heatmap", image)
        
        # Write image to video
        out.write(image)
        
        print(filename)

        #cv2.waitKey(0)

    out.release()
    
    # closing all open windows
    cv2.destroyAllWindows() 
    
    
if __name__ == "__main__":
    args = parse_arguments()

    main(*args)
