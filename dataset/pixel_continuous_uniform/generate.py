from PIL import Image, ImageDraw
from math import cos, sqrt
import random

# sequence name
path = "00" 

# Define the possible ball motions
motion_options = [
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1)
]

# Random the moving points
points = [(random.randint(1, 24), random.randint(1, 24)) for _ in range(10000)]

# Define the euclidean distance
distance = lambda p, l: sqrt((p[0]-l[0])**2 + (p[1]-l[1])**2)


def move(start_point=(0,0)):
    
    x, y = start_point
    
    for p in points:
        
        while distance(p, (x,y)) > 1:
            
            # Calc the distance for all possible motion options
            motion = [distance(p, (x+l[0], y+l[1])) for l in motion_options]
            
            argmin_motion = motion.index(min(motion))
            
            selected_motion = motion_options[argmin_motion]
            
            x, y = (x+selected_motion[0], y+selected_motion[1])
        
            yield x, y


with open(f"{path}/ground-truth.txt", "w") as ground_truth_file:
    
    i = 0
    
    for x, y in move():

        im = Image.new('RGB', (25, 25), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        # Draw the point
        draw.point((x, y), fill=(255, 255, 255))

        # Save the image
        im.save(f"{path}/images/{i}.jpg", quality=100) 
        
        i += 1
        
        # Print the ground truth (ball's center)
        print(f"{x},{y}", file=ground_truth_file)

