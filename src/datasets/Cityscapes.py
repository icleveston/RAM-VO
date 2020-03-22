import cv2
import os
from .Dataset import Dataset

class Cityscapes:

    def __init__(self, city, type='train'):

        # Define the basedir
        basedir = '/run/media/iury/Cleveston-2/datasets/cityscapes/leftImg8bit'

        self.path = os.path.join(basedir, type, city)

    def next_frame(self):

        # Foreach dir
        for _, _, dir in os.walk(self.path):
            for file in sorted(dir):
                if file.endswith(".png"):

                    # Open image
                    image = cv2.imread(os.path.join(self.path, file))

                    yield image
