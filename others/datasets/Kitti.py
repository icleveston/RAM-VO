import cv2
import os
import numpy as np
from .Dataset import Dataset

class Kitti(Dataset):

    def __init__(self, sequence, camera='image_0'):

        self.path_evaluation = '/run/media/iury/Cleveston-2/code/results/kitti'
        data_dir = '/run/media/iury/Cleveston-2/datasets/kitti_gray'

        self.path_frames = os.path.join(data_dir, 'sequences', sequence, camera)
        self.ground_truth = os.path.join(data_dir, 'poses', sequence + ".txt")

    def next_frame(self):

        # Foreach dir
        for _, _, dir in os.walk(self.path_frames):
            for file in sorted(dir):
                if file.endswith(".png"):

                    # Open image
                    image = cv2.imread(os.path.join(self.path_frames, file))

                    yield image

    def evaluate(self, evaluation_dir, plot=False, plot_mode='xz'):

        path = os.path.join(self.path_evaluation, evaluation_dir)

        plot_command = "-p" if plot else ""

        os.system(f"evo_traj kitti {path}/trajectory.txt --ref={self.ground_truth} {plot_command} --plot_mode {plot_mode} --save_plot {path}/plot_traj.pdf")
        os.system(f"evo_ape kitti {self.ground_truth} -va {plot_command} --plot_mode {plot_mode} {path}/trajectory.txt --save_results {path}/results_ape.zip --save_plot {path}/plot_ape.pdf")
        os.system(f"evo_rpe kitti {self.ground_truth} -va {plot_command} --plot_mode {plot_mode} {path}/trajectory.txt --save_results {path}/results_rpe.zip --save_plot {path}/plot_rpe.pdf")
