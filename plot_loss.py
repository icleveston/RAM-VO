import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import str2bool


def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", type=str, required=True, help="path to the output execution")
    arg.add_argument("--minibatch", type=str2bool, default=False, help="plot the losses by mini batch")
    arg.add_argument("--save", type=str2bool, default=False, help="save the plot")
    
    args = vars(arg.parse_args())
    
    return args["data_dir"], args["minibatch"], args["save"]


def main(data_dir, minibatch, save):
        
    basepath = os.path.join("out", data_dir, "loss")
        
    # Count the files inside the folder
    count_files = len(next(os.walk(basepath))[2])
    
    mse_array = []
    mae_array = []
    reward_array = []
    loss_action_array = []
    loss_baseline_array = []
    loss_reinforce_0_array = []
    loss_reinforce_1_array = []
    loss_all_array = []
    
    interations = None
    
    for i in range(1, count_files+1):
   
        # Build the loss path
        path = os.path.join("out", data_dir, "loss", f"loss_epoch_{i}.p")
                
        # Read the pickle files
        losses = pickle.load(open(path, "rb"))
        
        # Convert losses to numpy
        losses = map(np.asarray, losses)
        
        # Unpack the losses
        mse, mae, reward, loss_action, loss_baseline, loss_reinforce_0, loss_reinforce_1 = losses
        
        if interations is None:
            interations = len(mse)
            
        if not minibatch:
            mse = np.array(mse.sum()/interations)
            mae = np.array(mae.sum()/interations)
            reward = np.array(reward.sum()/interations)
            loss_action = np.array(loss_action.sum()/interations)
            loss_baseline = np.array(loss_baseline.sum()/interations)
            loss_reinforce_0 = np.array(loss_reinforce_0.sum()/interations)
            loss_reinforce_1 = np.array(loss_reinforce_1.sum()/interations)
            
        # Concat the losses for all epochs
        mse_array = np.concatenate((mse_array, mse), axis=None)
        mae_array = np.concatenate((mae_array, mae), axis=None)
        reward_array = np.concatenate((reward_array, reward), axis=None)
        loss_action_array = np.concatenate((loss_action_array, loss_action), axis=None)
        loss_baseline_array = np.concatenate((loss_baseline_array, loss_baseline), axis=None)
        loss_reinforce_0_array = np.concatenate((loss_reinforce_0_array, loss_reinforce_0), axis=None)
        loss_reinforce_1_array = np.concatenate((loss_reinforce_1_array, loss_reinforce_1), axis=None)
        loss_all_array = np.concatenate((loss_all_array, [loss_action + loss_baseline + loss_reinforce_0 + loss_reinforce_1]), axis=None)
    
    # Build the plot order array
    plot_order_array = [mse_array, mae_array, reward_array, loss_all_array, loss_action_array, loss_baseline_array, loss_reinforce_0_array, loss_reinforce_1_array]
    
    # Define the subplots titles
    titles = ['MSE', 'MAE', 'Reward', 'All Losses', 'Action Loss (MSE)', 'Baseline Loss (MSE)', 'Reinforce Loss 0', 'Reinforce Loss 1']

    # Create the plots
    fig, axs = plt.subplots(nrows=4, ncols=2)
    fig.set_size_inches(15, 15)
    fig.set_dpi(80)
    fig.tight_layout(rect=[0.005, 0, 1, 0.95], pad=2.0, w_pad=3.0, h_pad=3.0)
    
    type_loss = 'Minibatch' if minibatch else 'Epoch'
    
    # Set the figure title
    fig.suptitle(f"Losses by {type_loss}", fontsize=14)
    
    for i, ax in enumerate(axs.flat):
        
        # Plot the data
        ax.plot(plot_order_array[i])
        
        ax.set_title(titles[i])
        
        amax = np.amax(plot_order_array[i])
        amin = np.amin(plot_order_array[i])
        amax += amax*0.1
        
        major_ticks = np.arange(amin, amax, (amax-amin)/10)
        minor_ticks = np.arange(amin, amax, (amax-amin)/20)

        # Set the y ticks
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
                
        # Format the y tick labels
        ax.set_yticklabels(list(map(lambda x: "%.3f" % x, major_ticks)))
        
        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major', alpha=0.8)
        
    if not save:
        plt.show()
    else:
        plt.savefig(os.path.join("out", data_dir, 'loss.svg'), orientation='landscape', dpi=100)
    
if __name__ == "__main__":
    args = parse_arguments()

    main(*args)
