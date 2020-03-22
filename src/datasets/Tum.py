import cv2
import os
from .Dataset import Dataset

class Tum(Dataset):

    def __init__(self, sequence):
        self.sequence = sequence

        # Define the basedir
        basedir = '/run/media/iury/Cleveston-2/datasets/tum/monocular'

        self.path = os.path.join(basedir, sequence, 'images')

    def next_frame(self):

        # Foreach dir
        for _, _, dir in os.walk(self.path):
            for file in sorted(dir):
                if file.endswith(".jpg"):

                    # Open image
                    image = cv2.imread(os.path.join(self.path, file))

                    yield image

    def ground_truth(self):

        pass