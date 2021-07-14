from evo.core import lie_algebra, metrics
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import file_interface
from evo.tools import plot
from evo.tools.settings import SETTINGS
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
import copy
import csv
import os.path
import pprint
from kitti_odometry import KittiEvalOdom

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", type=str, required=True, help="path to the output execution")
   
    args = vars(arg.parse_args())
    
    return args["data_dir"]


def trajectory(path, groundtruth, predictions, mode=plot.PlotMode.xz):

    plot_collection = plot.PlotCollection("Trajectory")

    # Create teh plot figures
    fig_traj = plt.figure(figsize=tuple(SETTINGS.plot_figsize))
    fig_xyz, axarr_xyz = plt.subplots(3, sharex="col", figsize=tuple(SETTINGS.plot_figsize))
    fig_rpy, axarr_rpy = plt.subplots(3, sharex="col", figsize=tuple(SETTINGS.plot_figsize))

    ax_traj = plot.prepare_axis(fig_traj, mode)

    plot.traj(ax_traj, mode, groundtruth, 
              style=SETTINGS.plot_reference_linestyle,
              color=SETTINGS.plot_reference_color,
              label='Groundtruth',
              alpha=SETTINGS.plot_reference_alpha)

    plot.draw_coordinate_axes(ax_traj, groundtruth, mode, SETTINGS.plot_axis_marker_scale)
        
    plot.traj_xyz(
        axarr_xyz, groundtruth, style=SETTINGS.plot_reference_linestyle,
        color=SETTINGS.plot_reference_color, 
        label='Groundtruth',
        alpha=SETTINGS.plot_reference_alpha)

    plot.traj_rpy(
        axarr_rpy, groundtruth, style=SETTINGS.plot_reference_linestyle,
        color=SETTINGS.plot_reference_color, 
        label='Groundtruth',
        alpha=SETTINGS.plot_reference_alpha)

    color = ['darkblue', 'darkred', 'darkgreen']

    for (x, y), c in zip(predictions.items(), color):
        
        plot.traj(ax_traj, mode, y,
              style=SETTINGS.plot_trajectory_linestyle, 
              color=c,
              label=x, 
              alpha=SETTINGS.plot_trajectory_alpha)

        plot.draw_coordinate_axes(ax_traj, y, mode, SETTINGS.plot_axis_marker_scale)

        plot.traj_xyz(axarr_xyz, y, 
                      SETTINGS.plot_trajectory_linestyle,
                      color='darkblue',
                      label=x,
                      alpha=SETTINGS.plot_trajectory_alpha)
               
        plot.traj_rpy(axarr_rpy, y, 
                      SETTINGS.plot_trajectory_linestyle,
                      color='darkblue',
                      label=x,
                      alpha=SETTINGS.plot_trajectory_alpha)

    plot_collection.add_figure("trajectory", fig_traj)
    plot_collection.add_figure("xyz_view", fig_xyz)
    plot_collection.add_figure("rpy_view", fig_rpy)
    
    for name, fig in plot_collection.figures.items():
        
        dest = f"{path}/{name}.pdf"
        fig.tight_layout()
        fig.savefig(dest)


def main(data_dir):

    path = f"all_trajectories/{data_dir}/"
    
    prediction_file_ppo_256 = f"{path}/prediction_ppo_256.txt"
    prediction_file_ppo_1024 = f"{path}/prediction_ppo_1024.txt"
    prediction_file_reinforce = f"{path}/prediction_reinforce.txt"
    groundtruth_file = f"{path}/groundtruth_kitti.txt"

    groundtruth = file_interface.read_kitti_poses_file(groundtruth_file)
    prediction_ppo_256 = file_interface.read_kitti_poses_file(prediction_file_ppo_256)
    prediction_ppo_1024 = file_interface.read_kitti_poses_file(prediction_file_ppo_1024)
    prediction_reinforce = file_interface.read_kitti_poses_file(prediction_file_reinforce)
    
    len_gt = len(groundtruth.poses_se3)
    len_pred = len(prediction_ppo_256.poses_se3)
    
    # Cut the gt poses to match the prediction total
    groundtruth = PosePath3D(poses_se3=groundtruth.poses_se3[0:len(prediction_ppo_256.poses_se3)])
    
    print(f"Length Groundtruth: {len_gt}, Length Pred: {len_pred}")
    
    prediction_ppo_256_aligned = copy.deepcopy(prediction_ppo_256)
    prediction_ppo_256_aligned.align_origin(groundtruth)
    lie_algebra.sim3(*prediction_ppo_256_aligned.align(groundtruth, correct_scale=False, correct_only_scale=False, n=-1))
    
    prediction_ppo_1024_aligned = copy.deepcopy(prediction_ppo_1024)
    prediction_ppo_1024_aligned.align_origin(groundtruth)
    lie_algebra.sim3(*prediction_ppo_1024_aligned.align(groundtruth, correct_scale=False, correct_only_scale=False, n=-1))
    
    prediction_reinforce_aligned = copy.deepcopy(prediction_reinforce)
    prediction_reinforce_aligned.align_origin(groundtruth)
    lie_algebra.sim3(*prediction_reinforce_aligned.align(groundtruth, correct_scale=False, correct_only_scale=False, n=-1))

    predictions = {'reinforce': prediction_reinforce_aligned,
                   'ppo 1024': prediction_ppo_1024_aligned,
                   'ppo 256': prediction_ppo_256_aligned}

    # Plot the trajectory
    trajectory(path, groundtruth, predictions, mode=plot.PlotMode.xz)


if __name__ == "__main__":

    args = parse_arguments()

    main(args)
