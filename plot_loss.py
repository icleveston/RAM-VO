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
    
    train_mse_array = []
    train_mae_array = []
    train_reward_array = []
    train_loss_action_array = []
    train_loss_baseline_array = []
    train_loss_reinforce_array = []
    train_loss_all_array = []
    
    val_mse_array = []
    val_mae_array = []
    val_reward_array = []
    val_loss_action_array = []
    val_loss_baseline_array = []
    val_loss_reinforce_array = []
    val_loss_all_array = []
    
    interations_train = None
    
    for i in range(1, count_files+1):
   
        # Build the loss path
        path = os.path.join("out", data_dir, "loss", f"loss_epoch_{i}.p")
                
        # Read the pickle files
        losses = pickle.load(open(path, "rb"))
        
        # Separate between train and validation
        train_loss, val_loss = losses
        
        train_loss = list(train_loss)
        
        if np.shape(train_loss)[0] == 7:
        # Unpack the losses
            train_mae, train_reward, train_loss_reinforce, train_loss_baseline= train_loss
            val_mae, val_reward, val_loss_baseline, val_loss_reinforce, _ = val_loss
        else:
            train_mae, train_reward, train_loss_reinforce, train_loss_baseline = train_loss
            val_mae, val_reward, val_loss_baseline, val_loss_reinforce = val_loss

        if interations_train is None:
            interations_train = len(train_mae)
            interations_val = len(val_mae)
            
        if not minibatch:
            train_mae = np.array(train_mae.sum()/interations_train)
            train_reward = np.array(train_reward.sum()/interations_train)
            train_loss_baseline = np.array(train_loss_baseline.sum()/interations_train)
            train_loss_reinforce = np.array(train_loss_reinforce.sum()/interations_train)

            val_mae = np.array(val_mae.sum()/interations_val)
            val_reward = np.array(val_reward.sum()/interations_val)
            val_loss_baseline = np.array(val_loss_baseline.sum()/interations_val)
            val_loss_reinforce = np.array(val_loss_reinforce.sum()/interations_val)
            
        # Concat the losses for all epochs for train
        train_mae_array = np.concatenate((train_mae_array, train_mae), axis=None)
        train_reward_array = np.concatenate((train_reward_array, train_reward), axis=None)
        train_loss_baseline_array = np.concatenate((train_loss_baseline_array, train_loss_baseline), axis=None)
        train_loss_reinforce_array = np.concatenate((train_loss_reinforce_array, train_loss_reinforce), axis=None)
        train_loss_all_array = np.concatenate((train_loss_all_array, [train_loss_baseline + train_loss_reinforce]), axis=None)
        
        # Concat the losses for all epochs for validation
        val_mae_array = np.concatenate((val_mae_array, val_mae), axis=None)
        val_reward_array = np.concatenate((val_reward_array, val_reward), axis=None)
        val_loss_baseline_array = np.concatenate((val_loss_baseline_array, val_loss_baseline), axis=None)
        val_loss_reinforce_array = np.concatenate((val_loss_reinforce_array, val_loss_reinforce), axis=None)
        val_loss_all_array = np.concatenate((val_loss_all_array, [val_loss_baseline + val_loss_reinforce]), axis=None)
    
    mae_array = [train_mae_array, val_mae_array]
    reward_array = [train_reward_array, val_reward_array]
    loss_baseline_array = [train_loss_baseline_array, val_loss_baseline_array]
    loss_reinforce_array = [train_loss_reinforce_array, val_loss_reinforce_array]
    loss_all_array = [train_loss_all_array, val_loss_all_array]
    
    # Build the plot order array
    plot_order_array = [
        mae_array,
        reward_array
    ]
    
    # Define the subplots titles
    titles = ['MAE', 'Reward']
    
    for i in range(len(titles)):
        
        # Create the plots
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_dpi(80)
        fig.tight_layout(rect=[0.005, 0, 1, 0.95], pad=2.0, w_pad=3.0, h_pad=3.0)
        
        type_loss = 'Minibatch' if minibatch else 'Epoch'
    
        # Plot the train and validation data
        ax.plot(plot_order_array[i][0], label="Train")
        
        if not minibatch:
            ax.plot(plot_order_array[i][1], label="Validation")
        else:
            
            # Get only the train data
            plot_order_array[i] = plot_order_array[i][0]
        
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
        
        ax.legend()

        plt.savefig(os.path.join("out", data_dir, f'loss_{i}.pdf'), orientation='landscape', dpi=100, bbox_inches='tight', pad_inches=0)
        
    
if __name__ == "__main__":
    args = parse_arguments()

    main(*args)
    
    
