import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from datasets import Kitti, Tum, Euroc, Zurich, Cityscapes, Make3D


def main():

    # Create the kitti dataset
    k = Kitti(sequence="00")
    #k = Tum(sequence="sequence_01")
    #k = Euroc(sequence='MH_01_easy')
    #k = Zurich()
    #k = Cityscapes(city='aachen', type='train')

    k.evaluate('test_01', plot=False)



    # for frame in k.next_frame():
    #      cv2.imshow("frame", frame)
    #      cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()

def plot_make3D():

    k = Make3D(dataset='dataset_1', type='test')

    for image, depth in zip(k.next_image(), k.next_depth()):

        image = cv2.resize(image, (460, 345))

        cv2.imshow("image", image)

        d = np.asarray(depth['Position3DGrid'])

        d_resize = cv2.resize(d, (460, 345))

        cv2.imshow("depth", d_resize)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()

    #plot_make3D()


