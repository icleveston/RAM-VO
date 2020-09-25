import cv2
import os
import scipy.io


class Make3D():

    def __init__(self, dataset='dataset_1', type='train'):

        # Define the basedir
        basedir = '/run/media/iury/Cleveston-2/datasets/make3D'

        self.path_image = os.path.join(basedir, dataset, type, 'image')
        self.path_depth = os.path.join(basedir, dataset, type, 'depth')

    def next_image(self):

        # Foreach dir
        for _, _, dir in os.walk(self.path_image):
            for file in sorted(dir):
                if file.endswith(".jpg"):

                    # Open image
                    image = cv2.imread(os.path.join(self.path_image, file))

                    yield image

    def next_depth(self):

        # Foreach dir
        for _, _, dir in os.walk(self.path_depth):
            for file in sorted(dir):
                if file.endswith(".mat"):
                    # Open image
                    mat = scipy.io.loadmat(os.path.join(self.path_depth, file))

                    yield mat
