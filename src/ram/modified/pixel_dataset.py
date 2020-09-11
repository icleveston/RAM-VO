#!/usr/bin/env python
import os
import cv2
import struct
from datetime import datetime
import numpy as np 


class Pixel:
    
    def __init__(self, name, batch_size=128, image_amount=1966):
  
        assert name in ['train', 'test', 'val']
        
        data_dir = 'dataset/pixel/00/'
        
        self.path_frames = os.path.join(data_dir, 'images')
        self.ground_truth = os.path.join(data_dir, "ground-truth.txt")
        
        self.content = []
        
        # Read the ground-truth lines
        with open(self.ground_truth) as f:
            self.content = f.readlines()
        
        # Remove the \n
        self.content = [x.strip() for x in self.content] 
        
        self._epochs_completed = 0
        self._index = 0
        self._batch_size = batch_size
        self._image_amount = image_amount

    def next_batch(self):  
        
        batch_x = []
        batch_y = []

        # Check the sizes
        assert self._batch_size <= self._image_amount, f"batch_size {self._batch_size} cannot be larger than data size {self._image_amount}"
       
        # Define the indexes to cut
        start = self._index
        self._index += self._batch_size
        end = self._index
        
        # If it has reached the end, start again
        if end > self._image_amount:
            self._epochs_completed += 1
            self._index = 0
            end = self._image_amount
        
        # Read the frames
        for i in range(start, end):
            
            try:
            
                # Open image
                image_t = cv2.imread(os.path.join(self.path_frames, f"{i}.jpg"))
                image_t_1 = cv2.imread(os.path.join(self.path_frames, f"{i+1}.jpg"))
                
                # Concat the two images
                image = cv2.hconcat([image_t, image_t_1])
                
                # Add to the x array
                batch_x.append(image)
                
            except:
                pass
        
        # Read the ground-truth          
        lines = self.content[start:end]  
            
        for l in lines:
            
            coordinates = l.split(',')
            
            # Convert the coordinates to int
            x = int(coordinates[0])
            y = int(coordinates[1])
            
            # Add to the x array
            batch_y.append(x)
            
        return batch_x, batch_y

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epochs_completed(self):
        return self._epochs_completed



if __name__ == '__main__':
    import numpy as np
    
    c = Pixel('train')
    
    while True:
        
        print(c.epochs_completed)
        x, y = c.next_batch()
        print(np.shape(x))
