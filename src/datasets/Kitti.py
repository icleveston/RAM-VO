import cv2
import os
from .Dataset import Dataset

class Kitti(Dataset):

    def __init__(self, sequence, camera='image_0'):
        self.sequence = sequence
        self.camera = camera

        # Define the basedir
        basedir = '/run/media/iury/Cleveston-2/datasets/kitti_gray/sequences'

        self.path = os.path.join(basedir, sequence, camera)

    def next_frame(self):

        # Foreach dir
        for _, _, dir in os.walk(self.path):
            for file in sorted(dir):
                if file.endswith(".png"):

                    # Open image
                    image = cv2.imread(os.path.join(self.path, file))

                    yield image

    def ground_truth(self):

        pass
    