import cv2
import os
from .Dataset import Dataset

class Zurich(Dataset):

    def __init__(self):

        # Define the basedir
        basedir = '/run/media/iury/Cleveston-2/datasets/zurich_urban_mav/data/MAV Images'

        self.path = os.path.join(basedir)

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