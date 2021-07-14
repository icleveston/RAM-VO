from evo.core import lie_algebra, metrics
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import file_interface
from evo.tools import plot
from evo.tools.settings import SETTINGS
import matplotlib.pyplot as plt
import argparse
import numpy as np
import copy
import csv
import os.path
import pprint
from kitti_odometry import KittiEvalOdom


def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", type=str, required=True, help="path to the output execution")
   
    args = vars(arg.parse_args())
    
    return args["data_dir"]


def trajectory(path, prediction, groundtruth, mode=plot.PlotMode.xz):

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

    plot.traj(ax_traj, mode, prediction,
              style=SETTINGS.plot_trajectory_linestyle, 
              color='darkblue',
              label='Prediction', 
              alpha=SETTINGS.plot_trajectory_alpha)

    plot.draw_coordinate_axes(ax_traj, prediction, mode, SETTINGS.plot_axis_marker_scale)
        
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

    plot.traj_xyz(axarr_xyz, prediction, 
                  SETTINGS.plot_trajectory_linestyle,
                  color='darkblue',
                  label='Prediction',
                  alpha=SETTINGS.plot_trajectory_alpha)
           
    plot.traj_rpy(axarr_rpy, prediction, 
                  SETTINGS.plot_trajectory_linestyle,
                  color='darkblue',
                  label='Prediction',
                  alpha=SETTINGS.plot_trajectory_alpha)

    plot_collection.add_figure("trajectory", fig_traj)
    plot_collection.add_figure("xyz_view", fig_xyz)
    plot_collection.add_figure("rpy_view", fig_rpy)
    
    for name, fig in plot_collection.figures.items():
        
        dest = f"{path}/{name}.pdf"
        fig.tight_layout()
        fig.savefig(dest)


def ape(path, prediction, groundtruth, mode=plot.PlotMode.xz):

    pose_relation = metrics.PoseRelation.translation_part
    ape_metric = metrics.APE(pose_relation)

    data = (groundtruth, prediction)

    ape_metric.process_data(data)

    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)

    ape_stats = ape_metric.get_all_statistics()

    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot.PlotMode.xz)

    plot.traj(ax, mode, groundtruth, '--', "gray", "Groundtruth")
    plot.traj_colormap(ax, prediction, 
                       ape_metric.error, 
                       mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
    ax.legend()
    dest = f"{path}/ape_error.pdf"
    fig.tight_layout()
    fig.savefig(dest)
    return ape_stat


def rpe(path, prediction, groundtruth, pose_relation, delta):

    data = (groundtruth, prediction)

    # normal mode
    delta_unit = metrics.Unit.frames

    # all pairs mode
    all_pairs = False 

    rpe_metric = metrics.RPE(pose_relation, delta, delta_unit, all_pairs)
    rpe_metric.process_data(data)

    rpe_stat = rpe_metric.get_statistic(metrics.StatisticsType.rmse)

    return rpe_stat


def process_kitti(path, groundtruth_file, prediction_file):

    groundtruth = file_interface.read_kitti_poses_file(groundtruth_file)
    prediction = file_interface.read_kitti_poses_file(prediction_file)
    
    len_gt = len(groundtruth.poses_se3)
    len_pred = len(prediction.poses_se3)
    
    print(f"Length Groundtruth: {len_gt}, Length Pred: {len_pred}")

    # Cut the gt poses to match the prediction total
    groundtruth = PosePath3D(poses_se3=groundtruth.poses_se3[0:len(prediction.poses_se3)])

    prediction_aligned = copy.deepcopy(prediction)
    prediction_aligned.align_origin(groundtruth)
    lie_algebra.sim3(*prediction_aligned.align(groundtruth, correct_scale=False, correct_only_scale=False, n=-1))

    # Plot the trajectory
    trajectory(path, prediction, groundtruth, mode=plot.PlotMode.xz)
    ape_stat = ape(path, prediction, groundtruth, mode=plot.PlotMode.xz)

    rpe_rot_error_array = []
    rpe_trans_error_array = []
    
    # Avoid error when trajectory is short
    try:

        for m in [100, 200, 300, 400, 500, 600, 700, 800]:
            
            rpe_rot_error = rpe(path, prediction, groundtruth, pose_relation=metrics.PoseRelation.rotation_angle_deg, delta=m) / m
            rpe_rot_error_array.append(rpe_rot_error)
            
            rpe_trans_error = rpe(path, prediction, groundtruth, pose_relation=metrics.PoseRelation.translation_part, delta=m) / m
            rpe_trans_error_array.append(rpe_trans_error)
            
    except:
        pass
        
    rpe_rot = sum(rpe_rot_error_array)/len(rpe_rot_error_array) * 100
    rpe_trans = sum(rpe_trans_error_array)/len(rpe_trans_error_array) * 100

    return ape_stat, rpe_rot, rpe_trans, len_pred, len_gt


def process_euroc(path, groundtruth_file, prediction_file):

    groundtruth = file_interface.read_euroc_csv_trajectory(groundtruth_file)
    prediction = file_interface.read_euroc_csv_trajectory(prediction_file)
        
    groundtruth = PosePath3D(positions_xyz=groundtruth.positions_xyz[0:len(prediction.positions_xyz)],
                             orientations_quat_wxyz=groundtruth.orientations_quat_wxyz[0:len(prediction.orientations_quat_wxyz)])
    
    prediction = PosePath3D(positions_xyz=prediction.positions_xyz,
                            orientations_quat_wxyz=prediction.orientations_quat_wxyz)

    prediction_aligned = copy.deepcopy(prediction)
    prediction_aligned.align_origin(groundtruth)
    lie_algebra.sim3(*prediction_aligned.align(groundtruth, correct_scale=False, correct_only_scale=False, n=-1))

    # Plot the trajectory
    trajectory(path, prediction, groundtruth, mode=plot.PlotMode.xy)
    ape_stat = ape(path, prediction, groundtruth, mode=plot.PlotMode.xy)

    # Compute the RPE and ATE for every 1 meter
    rpe_rot = rpe(path, prediction, groundtruth, pose_relation=metrics.PoseRelation.rotation_angle_deg, delta=1) / 1
    rpe_trans = rpe(path, prediction, groundtruth, pose_relation=metrics.PoseRelation.translation_part, delta=1) / 1 * 100

    return ape_stat, rpe_rot, rpe_trans, len_pred, len_gt


def main(data_dir):

    path = f"out/{data_dir}/"
    
    prediction_file = f"{path}/prediction.txt"
    kitti_groundtruth_file = f"{path}/groundtruth_kitti.txt"
    euroc_groundtruth_file = f"{path}/groundtruth_euroc.csv"

    if os.path.isfile(kitti_groundtruth_file):
        ape_stat, rpe_rot, rpe_trans, len_pred, len_gt = process_kitti(path, kitti_groundtruth_file, prediction_file)
    else:
        ape_stat, rpe_rot, rpe_trans, len_pred, len_gt = process_euroc(path, euroc_groundtruth_file, prediction_file)

    eval_tool = KittiEvalOdom()
<<<<<<< HEAD
    rpe_trans, rpe_rot, ape_stat, _, _ = eval_tool.eval(data_dir, alignment='6dof')
=======
    rpe_trans, rpe_rot, ape_stat, _, _ = eval_tool.eval(data_dir)
>>>>>>> 261768f80a (plot optical flow with glimspes)
        
    # Save the metrics
    with open(f"{path}/stats.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
        writer.writeheader()
        writer.writerow({'metric': 'Groundtruth Length', 'value': len_gt})
        writer.writerow({'metric': 'Prediction Length', 'value': len_pred})
        writer.writerow({'metric': 'RPE trans (%)', 'value': rpe_trans})
        writer.writerow({'metric': 'RPE rot (deg/100m)', 'value': rpe_rot})
        writer.writerow({'metric': 'ATE (m)', 'value': ape_stat})


if __name__ == "__main__":

    args = parse_arguments()

    main(args)
