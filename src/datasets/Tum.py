import cv2
import os
from .Dataset import Dataset


class Tum(Dataset):

    def __init__(self, sequence):
        self.sequence = sequence

        # Define the basedir
        self.path_evaluation = '/run/media/iury/Cleveston-2/code/results/tum'
        data_dir = '/run/media/iury/Cleveston-2/datasets/tum/monocular'

        self.path_frames = os.path.join(data_dir, sequence)
        self.ground_truth = os.path.join(data_dir, sequence, "groundtruthSync.txt")

    def next_frame(self):

        # Foreach dir
        for _, _, dir in os.walk(self.path_frames):
            for file in sorted(dir):
                if file.endswith(".jpg"):
                    # Open image
                    image = cv2.imread(os.path.join(self.path_frames, file))

                    yield image

    def evaluate(self, evaluation_dir, plot=False, plot_mode='xz'):

        path = os.path.join(self.path_evaluation, evaluation_dir)

        plot_command = "-p" if plot else ""

        os.system(
            f"evo_traj tum {path}/trajectory.txt --ref={self.ground_truth} {plot_command} --plot_mode {plot_mode} --save_plot {path}/plot_traj.pdf")
        os.system(
            f"evo_ape tum {self.ground_truth} -va {plot_command} --plot_mode {plot_mode} {path}/trajectory.txt --save_results {path}/results_ape.zip --save_plot {path}/plot_ape.pdf")
        os.system(
            f"evo_rpe tum {self.ground_truth} -va {plot_command} --plot_mode {plot_mode} {path}/trajectory.txt --save_results {path}/results_rpe.zip --save_plot {path}/plot_rpe.pdf")
