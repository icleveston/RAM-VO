import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", type=str, required=True, help="path to the output execution")
    arg.add_argument("--minibatch", type=str2bool, default=False, help="plot the losses by mini batch")
    
    args = vars(arg.parse_args())
    
    return args["data_dir"], args["minibatch"]


def main(data_dir, minibatch):
        
    # Count the files inside the folder
    count_files = len(next(os.walk(os.path.join("out", data_dir, "loss")))[2])
    
    accuracy_array = []
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
        accuracy, loss_action, loss_baseline, loss_reinforce_0, loss_reinforce_1 = losses
        
        if interations is None:
            interations = len(accuracy)
            
        if not minibatch:
            accuracy = np.array(accuracy.sum()/interations)
            loss_action = np.array(loss_action.sum()/interations)
            loss_baseline = np.array(loss_baseline.sum()/interations)
            loss_reinforce_0 = np.array(loss_reinforce_0.sum()/interations)
            loss_reinforce_1 = np.array(loss_reinforce_1.sum()/interations)
            
        # Concat the losses for all epochs
        accuracy_array = np.concatenate((accuracy_array, accuracy), axis=None)
        loss_action_array = np.concatenate((loss_action_array, loss_action), axis=None)
        loss_baseline_array = np.concatenate((loss_baseline_array, loss_baseline), axis=None)
        loss_reinforce_0_array = np.concatenate((loss_reinforce_0_array, loss_reinforce_0), axis=None)
        loss_reinforce_1_array = np.concatenate((loss_reinforce_1_array, loss_reinforce_1), axis=None)
        loss_all_array = np.concatenate((loss_all_array, [loss_action + loss_baseline + loss_reinforce_0 + loss_reinforce_1]), axis=None)
    
    # Build the plot order array
    plot_order_array = [accuracy_array, loss_all_array, loss_action_array, loss_baseline_array, loss_reinforce_0_array, loss_reinforce_1_array]
    
    # Define the subplots titles
    titles = ['Accuracy', 'Loss', 'Action Loss', 'Baseline Loss', 'Reinforce Loss 0', 'Reinforce Loss 1']

    # Create the plots
    fig, axs = plt.subplots(nrows=3, ncols=2)
    fig.set_dpi(300)
    
    type_loss = 'Minibatch' if minibatch else 'Epoch'
    
    # Set the figure title
    fig.suptitle(f"Losses by {type_loss}", fontsize=14)
    
    for i, ax in enumerate(axs.flat):
        
        # Plot the data
        ax.plot(plot_order_array[i])
        
        ax.set_title(titles[i])
        ax.grid(True)
    
    plt.show()
    
if __name__ == "__main__":
    args = parse_arguments()

    main(*args)
