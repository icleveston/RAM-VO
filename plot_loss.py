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
    
    train_loss_regressor_array = []
    train_loss_rot_array = []
    train_loss_tran_array = []
    train_reward_array = []
    train_loss_reinforce_array = []
    train_loss_baseline_array = []
    
    val_loss_regressor_array = []
    val_loss_rot_array = []
    val_loss_tran_array = []
    val_reward_array = []
    val_loss_reinforce_array = []
    val_loss_baseline_array = []
    
    interations_train = None
    
    for i in range(1, count_files+1):
   
        # Build the loss path
        path = os.path.join("out", data_dir, "loss", f"loss_epoch_{i}.p")
                
        # Read the pickle files
        losses = pickle.load(open(path, "rb"))
        
        # Separate between train and validation
        train_loss, val_loss = losses
        
        # Unpack the losses
        train_reward, train_loss_regressor, train_loss_rot, train_loss_tran, train_loss_baseline, train_loss_reinforce = train_loss
        val_reward, val_loss_regressor, val_loss_rot, val_loss_tran, val_loss_baseline, val_loss_reinforce = val_loss

        if interations_train is None:
            interations_train = len(train_loss_regressor)
            interations_val = len(val_loss_regressor)
            
        if not minibatch:
            train_reward = np.array(train_reward.sum()/interations_train)
            train_loss_regressor = np.array(train_loss_regressor.sum()/interations_train)
            train_loss_rot = np.array(train_loss_rot.sum()/interations_train)
            train_loss_tran = np.array(train_loss_tran.sum()/interations_train)
            train_loss_baseline = np.array(train_loss_baseline.sum()/interations_train)
            train_loss_reinforce = np.array(train_loss_reinforce.sum()/interations_train)

            val_reward = np.array(val_reward.sum()/interations_val)
            val_loss_regressor = np.array(val_loss_regressor.sum()/interations_val)
            val_loss_rot = np.array(val_loss_rot.sum()/interations_val)
            val_loss_tran = np.array(val_loss_tran.sum()/interations_val)
            val_loss_baseline = np.array(val_loss_baseline.sum()/interations_val)
            val_loss_reinforce = np.array(val_loss_reinforce.sum()/interations_val)
            
        # Concat the losses for all epochs for train
        train_reward_array = np.concatenate((train_reward_array, train_reward), axis=None)
        train_loss_regressor_array = np.concatenate((train_loss_regressor_array, train_loss_regressor), axis=None)
        train_loss_rot_array = np.concatenate((train_loss_rot_array, train_loss_rot), axis=None)
        train_loss_tran_array = np.concatenate((train_loss_tran_array, train_loss_tran), axis=None)
        train_loss_baseline_array = np.concatenate((train_loss_baseline_array, train_loss_baseline), axis=None)
        train_loss_reinforce_array = np.concatenate((train_loss_reinforce_array, train_loss_reinforce), axis=None)
        
        # Concat the losses for all epochs for validation
        val_reward_array = np.concatenate((val_reward_array, val_reward), axis=None)
        val_loss_regressor_array = np.concatenate((val_loss_regressor_array, val_loss_regressor), axis=None)
        val_loss_rot_array = np.concatenate((val_loss_rot_array, val_loss_rot), axis=None)
        val_loss_tran_array = np.concatenate((val_loss_tran_array, val_loss_tran), axis=None)
        val_loss_baseline_array = np.concatenate((val_loss_baseline_array, val_loss_baseline), axis=None)
        val_loss_reinforce_array = np.concatenate((val_loss_reinforce_array, val_loss_reinforce), axis=None)
    
    reward_array = [train_reward_array, val_reward_array]
    loss_regressor_array = [train_loss_regressor_array, val_loss_regressor_array]
    loss_rot_array = [train_loss_rot_array, val_loss_rot_array]
    loss_tran_array = [train_loss_tran_array, val_loss_tran_array]
    loss_baseline_array = [train_loss_baseline_array, val_loss_baseline_array]
    loss_reinforce_array = [train_loss_reinforce_array, val_loss_reinforce_array]
    
    # Build the plot order array
    plot_order_array = [
        loss_regressor_array,
        loss_rot_array,
        loss_tran_array, 
        reward_array, 
        loss_reinforce_array,
        loss_baseline_array
    ]
    
    # Define the subplots titles
    titles = ['Regressor Loss (MSE)','Regressor Rotation (MSE)', 'Regressor Translation (MSE)', 'Reward', 'Reinforce Loss', 'Baseline Loss (MSE)']

    # Create the plots
    fig, axs = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(20, 10)
    fig.set_dpi(80)
    fig.tight_layout(rect=[0.03, 0, 1, 0.95], pad=2.0, w_pad=4.0, h_pad=3.0)
    
    type_loss = 'Minibatch' if minibatch else 'Epoch'
    
    # Set the figure title
    fig.suptitle(f"Losses by {type_loss}", fontsize=14)
    
    for i, ax in enumerate(axs.flat):      

        if not minibatch:
            ax.plot(plot_order_array[i][0][1:], label="Train")
            ax.plot(plot_order_array[i][1][1:], label="Validation")
            
            plot_order_array_np = np.asarray(plot_order_array[i])[:, 1:].reshape(-1)
            
            amax = np.amax(plot_order_array_np)
            amin = np.amin(plot_order_array_np)
        else:

            ax.plot(plot_order_array[i][0][100:], label="Train")
            amax = np.amax(plot_order_array[i][0][100:])
            amin = np.amin(plot_order_array[i][0][100:])

        ax.set_title(titles[i])

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

        ax.legend()
        
    if not save:
        plt.show()
    else:
        
        name = 'minibatch' if minibatch else 'epoch'
        
        plt.savefig(os.path.join("out", data_dir, f'loss_{name}.pdf'), orientation='landscape', dpi=100)
    
if __name__ == "__main__":
    args = parse_arguments()

    main(*args)
