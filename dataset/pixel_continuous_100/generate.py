from PIL import Image, ImageDraw
from math import cos, sqrt
import random

# sequence name
path = "00" 

# How many points to generate
points_count = 5000

# Set the minimum distance between points
min_point_distance = 50

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

points = []

# Generate the points
while points_count > 0:
    
    # Random the point
    point = (random.randint(0, 98), random.randint(0, 98))
    
    # If the distance from the previous points is greater
    if len(points) == 0 or (distance(point, points[-1]) > min_point_distance) :
        
        points.append(point)
    
        points_count -= 1
        
        
def move(start_point=(0,0)):
    
    x, y = start_point
    
    # For each point
    for p in points:
        
        # While not reached
        while distance(p, (x,y)) > 0:
            
            # Calc the distance for all possible motion options
            motion = [distance(p, (x+l[0], y+l[1])) for l in motion_options]
        
            argmin_motion = motion.index(min(motion))
            
            # Select the correct direction to move
            selected_motion = motion_options[argmin_motion]
            
            # Set how many pixels to skip
            skipped = random.randint(0, 5)
            
            # Avoid to go outside the board
            if x+selected_motion[0]*skipped < 0 or x+selected_motion[0]*skipped > 99 or y+selected_motion[1]*skipped < 0 or y+selected_motion[1]*skipped > 99: continue
            
            x, y = (x+selected_motion[0]*skipped, y+selected_motion[1]*skipped)
            
            yield x, y
               

with open(f"{path}/ground-truth.txt", "w") as ground_truth_file:
    
    i = 0
    
    for x, y in move():

        # Create the image
        im = Image.new('RGB', (100, 100), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        # Draw the point
        draw.point((x, y), fill=(255, 255, 255))
        draw.point((x+1, y), fill=(255, 255, 255))
        draw.point((x, y+1), fill=(255, 255, 255))
        draw.point((x+1, y+1), fill=(255, 255, 255))

        # Save the image
        im.save(f"{path}/images/{i}.jpg", quality=100) 
        
        i += 1
        
        # Print the ground truth (ball's center)
        print(f"{x},{y}", file=ground_truth_file)

