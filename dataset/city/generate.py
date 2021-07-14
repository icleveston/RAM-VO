from PIL import Image, ImageDraw
from math import cos, sqrt
import random
import math

random.seed(1)

sign = lambda x: math.copysign(1, x)

# sequence name
path = "../../../dataset/city" 

# The image dimensions
img_w = 6000 - 300
img_h = 4000 - 300

# How many points to generate
points_count = 200

# Set the minimum distance between points
min_point_distance = 3000

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

# Define the euclidean distance
distance = lambda p, l: abs(p[0]-l[0]) + abs(p[1]-l[1])

# The first point in the middle ofthe image
points = [[img_w//2, img_h//2]]

# Generate the points
while points_count > 0:
    
    # Random the point
    point = (random.randint(0, img_w), random.randint(0, img_h))
    
    # If the distance from the previous points is greater
    if distance(point, points[-1]) > min_point_distance:
        
        points.append(point)
    
        points_count -= 1
     
        
def move():
    
    x, y = points[0]
    
    # For each point
    for p in points[1:]:
        
        # While not reached
        while distance(p, (x,y)) > 0:
            
            # Calc the distance for all possible motion options
            motion = [distance(p, (x+l[0], y+l[1])) for l in motion_options]

            argmin_motion = motion.index(min(motion))

            # Select the correct direction to move
            selected_motion = motion_options[argmin_motion]

            # Set how many pixels to skip
            skipped_x = random.randint(0, 50)
            skipped_y = random.randint(0, 50)
            
            shift_x = x + selected_motion[0]*skipped_x
            shift_y = y + selected_motion[1]*skipped_y
            
            if shift_x < 0 or shift_x > img_w or shift_y < 0 or shift_y > img_h:
                continue
            elif distance(p, (shift_x, shift_y)) < 50:
                x, y = p
            else:
                x, y = shift_x, shift_y
            
            yield x, y

with open(f"{path}/ground-truth.txt", "w") as ground_truth_file:
    
    i = 0
    
    x_ant = None
    y_ant = None
    
    # Open the source image
    image_original = Image.open('source.jpg').convert("L")
    
    for (x, y) in move():
        
        if x_ant is None:
            x_ant = x
            y_ant = y
        
        # Crop the image
        image = image_original.crop((x, y, x+300, y+300))

        # Save the image
        image.save(f"{path}/images/{i}.jpg") 
        
        i += 1
        
        dx = int(x - x_ant)
        dy = int(y - y_ant)
        
        # Print the ground truth
        print(f"{dx},{dy}", file=ground_truth_file)
        
        x_ant = x
        y_ant = y


