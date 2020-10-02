from PIL import Image, ImageDraw
from math import cos, sqrt
import random

# sequence name
path = "00" 

# How many points to generate
points_count = 50000

# Set the minimum distance between points
min_point_distance = 10
max_point_distance = 15

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
    point = (random.randint(1, 24), random.randint(1, 24))
    
    # If the distance from the previous points is greater
    if len(points) == 0 or (distance(point, points[-1]) > min_point_distance and distance(point, points[-1]) < max_point_distance) :
        
        points.append(point)
    
        points_count -= 1
        
        
def move(start_point=(0,0)):
    
    x, y = start_point
    
    for p in points:
        
        # Set how many pixels to skip
        skipped = random.randint(1, 8)
        
        j = 0
        
        while distance(p, (x,y)) > 0:
            
            # Calc the distance for all possible motion options
            motion = [distance(p, (x+l[0], y+l[1])) for l in motion_options]
            
            argmin_motion = motion.index(min(motion))
            
            selected_motion = motion_options[argmin_motion]
            
            x, y = (x+selected_motion[0], y+selected_motion[1])
            
            if j in [0,2,3]:
                j += 1
                yield x, y
            elif j <= skipped+3:
                j += 1
            else:
                yield x, y
               

with open(f"{path}/ground-truth.txt", "w") as ground_truth_file:
    
    i = 0
    
    for x, y in move():

        # Create the image
        im = Image.new('RGB', (25, 25), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        # Draw the point
        draw.point((x, y), fill=(255, 255, 255))

        # Save the image
        im.save(f"{path}/images/{i}.jpg", quality=100) 
        
        i += 1
        
        # Print the ground truth (ball's center)
        print(f"{x},{y}", file=ground_truth_file)


