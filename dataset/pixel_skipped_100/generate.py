from PIL import Image, ImageDraw
from math import cos, sqrt
import random

# sequence name
path = "00" 

# How many points to generate
points_count = 100000

# Set the minimum distance between points
min_point_distance = 0
max_point_distance = 99

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
    if len(points) == 0 or (distance(point, points[-1]) > min_point_distance and distance(point, points[-1]) < max_point_distance) :
        
        points.append(point)
    
        points_count -= 1

with open(f"{path}/ground-truth.txt", "w") as ground_truth_file:
    
    i = 0
    
    for (x, y) in points:

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


