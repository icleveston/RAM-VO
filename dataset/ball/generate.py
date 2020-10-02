from PIL import Image, ImageDraw
from math import cos, sqrt
import random

# sequence name
path = "00" 

# ball radius
r = 15 

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
points = [(random.randint(40, 260), random.randint(40, 260)) for _ in range(10000)]

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

        im = Image.new('RGB', (300, 300), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        # Draw the ball
        draw.ellipse((x, y, x+2*r, y+2*r), fill=(255, 0, 0), outline=(255, 0, 0))
        
        #draw.rectangle((200, 100, 300, 200), fill=(0, 192, 192), outline=(255, 255, 255))
        #draw.line((350, 200, 450, 100), fill=(255, 255, 0), width=10)

        # Save the image
        im.save(f"{path}/images/{i}.jpg", quality=100) 
        
        i += 1
        
        # Print the ground truth (ball's center)
        print(f"{x+r},{y+r}", file=ground_truth_file)

