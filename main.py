import os
import time
import shutil
import pickle
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model import RecurrentAttention
from utils import *
from data_loader import get_data_loader


class Main:
    
    def __init__(self, test=None, resume=None):
        
        # Glimpse Network Params
        self.patch_size = 16 # size of extracted patch at highest res
        self.glimpse_scale = 2 # scale of successive patches
        self.num_patches = 3 # number of downscaled patches per glimpse
        self.loc_hidden = 128 # hidden size of loc fc
        self.glimpse_hidden = 128 # hidden size of glimpse fc

        # Core Network Params
        self.num_glimpses = 4 # number of glimpses, i.e. BPTT iterations
        self.hidden_size = 512 # hidden size of rnn

        # Reinforce Params
        self.std = 0.05 # gaussian policy standard deviation
        self.M = 1 # Monte Carlo sampling for valid and test sets
        
        # Data Params
        self.valid_size = 0.1 # Proportion of training set used for validation
        self.test_size = 0.05 # Proportion of training set used for test
        self.batch_size = 128 # number of images in each batch of data
        self.num_workers = 4 # number of subprocesses to use for data loading
        self.num_out = 2
        self.num_channels = 3
            
        # Training params
        self.epochs = 200 # number of epochs to train for
        self.start_epoch = 0
        self.momentum = 0.5 # Nesterov momentum value
        self.lr = 3e-4 # Initial learning rate value
        self.lr_patience = 20 # Number of epochs to wait before reducing lr
        self.train_patience = 50 # Number of epochs to wait before stopping train

        # Other params
        self.random_seed = 1 # Seed to ensure reproducibility
        self.best = True # Load best model or most recent for testing
        self.print_freq = 10 # How frequently to print training details
        self.num_workers = 4
        self.pin_memory = False
        self.best_valid_acc = 10000000.0
        self.counter = 0
        
        # Set the seed
        torch.manual_seed(self.random_seed)
        
         # Check if the gpu is available
        if torch.cuda.is_available():
            
            self.device = torch.device("cuda")
            
            torch.cuda.manual_seed(self.random_seed)
            
            self.num_workers = 1
            self.pin_memory = True
            
        else:
            self.device = torch.device("cpu")

        print(f"\n[*] Device: {self.device}")
        
        # Build the model
        self.model = RecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.loc_hidden,
            self.glimpse_hidden,
            self.std,
            self.hidden_size,
            self.num_out,
        )
        
        # Set the model to the device
        self.model.to(self.device)

        # Start the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Start the scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=self.lr_patience)
           
        # Set the data loader
        self.train_loader, self.valid_loader, self.test_loader = get_data_loader(
            self.batch_size,
            self.valid_size,
            self.test_size,
            self.num_workers,
            self.pin_memory
        )
        
        self.num_train = len(self.train_loader.sampler)
        self.num_valid = len(self.valid_loader.sampler)
        self.num_test = len(self.test_loader.sampler)
            
        # Execute the training or testing
        if test is None:
            
            # Should resume the train
            if resume is None:
                
                # Set the folder time for each execution
                folder_time = time.strftime("%Y_%m_%d_%H_%M_%S")

                # Set the model name
                self.model_name = f"exec_{self.num_glimpses}_{self.patch_size}_{self.num_patches}_{self.glimpse_scale}_{folder_time}"

                # Set the folders
                self.output_path = os.path.join('out', self.model_name)
                self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
                self.glimpse_path = os.path.join(self.output_path, 'glimpse')
                self.loss_path = os.path.join(self.output_path, 'loss')
                
                # Create the folders
                if not os.path.exists(self.output_path):
                    os.makedirs(self.output_path)
                if not os.path.exists(self.checkpoint_path):
                    os.makedirs(self.checkpoint_path)
                if not os.path.exists(self.glimpse_path):
                    os.makedirs(self.glimpse_path)
                if not os.path.exists(self.loss_path):
                    os.makedirs(self.loss_path)
            
            else:
                
                # Set the model to be loaded
                self.model_name = resume
                
                # Set the folders
                self.output_path = os.path.join('out', self.model_name)
                self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
                self.glimpse_path = os.path.join(self.output_path, 'glimpse')
                self.loss_path = os.path.join(self.output_path, 'loss')
                
                # Load the model
                self._load_checkpoint(best=False)            
            
            print(f"[*] Output Folder: {self.model_name}")
            
            # Print the model info
            count_parameters(self.model, print_table=False)
            
            # Save the configuration as image
            self._save_config()
            
            self.train()
            
        else:
            
            # Set the model to load
            self.model_name = test
            
            # Set the folders
            self.output_path = os.path.join('out', self.model_name)
            self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
            self.glimpse_path = os.path.join(self.output_path, 'glimpse')
            self.loss_path = os.path.join(self.output_path, 'loss')
            
            # Load the model
            self._load_checkpoint(best=False)  
            
            # Print the model info
            count_parameters(self.model, print_table=False)
            
            self.test()            

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        print(f"[*] Train on {self.num_train} samples, validate on {self.num_valid} samples")

        # For each epoch
        for epoch in range(self.start_epoch, self.epochs):

            # Get the current lr
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"\nEpoch: {epoch+1}/{self.epochs} - LR: {current_lr}")

            # Train one epoch
            train_loss, train_mse, train_mae = self._train_one_epoch(epoch)

            # Validate one epoch
            valid_loss, valid_mse, valid_mae = self.validate(epoch)

            # Reduce lr if validation loss plateaus
            self.scheduler.step(valid_mae)

            # Check if it is the best model
            is_best = valid_mse < self.best_valid_acc
            
            msg = "train loss: {:.3f}, mse: {:.3f}, mae: {:.3f} | val loss: {:.3f}, mse: {:.3f}, mae: {:.3f}"
            
            # Check for improvement
            if is_best:
                self.counter = 0
                msg += " [*]"
            else:
                self.counter += 1
            
            print(msg.format(train_loss, train_mse, train_mae, valid_loss, valid_mse, valid_mae))
                
            #if self.counter > self.train_patience:
            #    print("[!] No improvement in a while, stopping training.")
            #    return
            
            self.best_valid_acc = min(valid_mse, self.best_valid_acc)
            
            # Save the checkpoint for each epoch
            self._save_checkpoint({
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                }, is_best
            )
            
        # Save the results as image
        self._save_results()

    def _train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        
        self.model.train()
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        mse_bar = AverageMeter()
        mae_bar = AverageMeter()
        
        # Store the losses array
        loss_action_array = []
        loss_baseline_array = []
        loss_reinforce_0_array = []
        loss_reinforce_1_array = []
        mse_array = []
        mae_array = []
        reward_array = []

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            
            # For each batch
            for i, (x, y) in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()
                
                # Set data to the respected device
                x_0, x_1, y = x[0].to(self.device), x[1].to(self.device), y.to(self.device)

                # initialize location vector and hidden state
                self.batch_size = x_0.shape[0]

                glimpse_location_0 = []
                glimpse_location_1 = []
                log_pi_0 = []
                log_pi_1 = []
                baselines = []
                predicted = None
                
                # Reset the model parameters
                h_state, c_state, l_t_0, l_t_1 = self._reset()
                
                # For each glimpse
                for t in range(self.num_glimpses):
                    
                    # Get the prediction on the last glimpse
                    is_last = t==self.num_glimpses-1
                    
                    # Call the model
                    h_state, l_t_0, l_t_1, b_t, predicted, p_0, p_1 = self.model(x_0, x_1, l_t_0, l_t_1, h_state, c_state, last=is_last)
    
                    # Store the glimpse location for both frames
                    glimpse_location_0.append(l_t_0)
                    glimpse_location_1.append(l_t_1)
                    
                    # Get the glimpse location loss
                    log_pi_0.append(p_0)
                    log_pi_1.append(p_1)
                    
                    # Add the baseline
                    baselines.append(b_t)

                # Convert list to tensors and reshape
                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi_0 = torch.stack(log_pi_0).transpose(1, 0)
                log_pi_1 = torch.stack(log_pi_1).transpose(1, 0)
                
                # Denormalize the predictions
                predicted_denormalized = torch.stack([denormalize_displacement(l, 100) for l in predicted])

                loss_mse = torch.nn.MSELoss()
                loss_l1 = torch.nn.L1Loss()
                
                # Compute the reward based on L1 norm
                R = 1/(1 + torch.sum(torch.abs(predicted_denormalized.detach() - y), dim=1))
                
                R = R.unsqueeze(1).repeat(1, self.num_glimpses)
                    
                # Compute losses for differentiable modules
                loss_action = loss_mse(predicted_denormalized, y)
                loss_baseline = loss_mse(baselines, R)

                # Compute reinforce loss, summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()

                loss_reinforce_0 = torch.sum(-log_pi_0 * adjusted_reward, dim=1)
                loss_reinforce_0 = torch.mean(loss_reinforce_0, dim=0) * 0.1
            
                loss_reinforce_1 = torch.sum(-log_pi_1 * adjusted_reward, dim=1)
                loss_reinforce_1 = torch.mean(loss_reinforce_1, dim=0) * 0.1

                # Join the losses
                loss = loss_action + loss_baseline + loss_reinforce_0 + loss_reinforce_1

                # Get the mse
                mse = loss_action
                mae = loss_l1(predicted_denormalized.detach(), y)
                
                # Store the losses
                loss_action_array.append(loss_action.cpu().data.numpy())
                loss_baseline_array.append(loss_baseline.cpu().data.numpy())
                loss_reinforce_0_array.append(loss_reinforce_0.cpu().data.numpy())
                loss_reinforce_1_array.append(loss_reinforce_1.cpu().data.numpy())
                mse_array.append(mse.cpu().data.numpy())
                mae_array.append(mae.cpu().data.numpy())
                reward_array.append(torch.mean(torch.sum(R, dim=1), dim=0).data.numpy())

                # Store the loss and metric
                losses.update(loss.item(), x_0.size()[0])
                mse_bar.update(mse.item(), x_0.size()[0])
                mae_bar.update(mae.item(), x_0.size()[0])

                # Compute gradients and update SGD
                loss.backward()
                self.optimizer.step()

                # Measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                # Set the var description
                pbar.set_description(("{:.1f}s - loss: {:.3f}, mse: {:.3f}, mae: {:.3f}".format((toc-tic), loss.item(), mse.item(), mae.item())))
                
                # Update the bar
                pbar.update(self.batch_size)

                # Save glimpses for the first batch
                if i == 0:
                    
                    # Format the data for storage
                    img_0 = [g.cpu().data.numpy().squeeze() for g in x_1]
                    img_1 = [g.cpu().data.numpy().squeeze() for g in x_0]
                    loc_0 = [l.cpu().data.numpy() for l in glimpse_location_0]
                    loc_1 = [l.cpu().data.numpy() for l in glimpse_location_1]
                    
                    # Build the data to be saved
                    data = ((img_0, loc_0), (img_1, loc_1))
                    
                    # Dump the glimpses
                    with open(os.path.join(self.glimpse_path, f"glimpses_epoch_{epoch+1}.p"), "wb") as f:
                        pickle.dump(data, f)

            # Dump the loss
            with open(os.path.join(self.loss_path, f"loss_epoch_{epoch+1}.p"), "wb") as f:
                
                data = (mse_array, mae_array, reward_array, loss_action_array, loss_baseline_array, loss_reinforce_0_array, loss_reinforce_1_array)
                
                pickle.dump(data, f)

            return losses.avg, mse_bar.avg, mae_bar.avg

    @torch.no_grad()
    def validate(self, epoch):
        """Evaluate the RAM model on the validation set.
        """
        losses = AverageMeter()
        mse_bar = AverageMeter()
        mae_bar = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
                
            self.optimizer.zero_grad()
            
            # Set data to the respected device
            x_0, x_1, y = x[0].to(self.device), x[1].to(self.device), y.to(self.device)

            # initialize location vector and hidden state
            self.batch_size = x_0.shape[0]

            glimpse_location_0 = []
            glimpse_location_1 = []
            log_pi_0 = []
            log_pi_1 = []
            baselines = []
            predicted = None
            
            # Reset the model parameters
            h_state, c_state, l_t_0, l_t_1 = self._reset()
            
            # For each glimpse
            for t in range(self.num_glimpses):
                
                # Get the prediction on the last glimpse
                is_last = t==self.num_glimpses-1
                
                # Call the model
                h_state, l_t_0, l_t_1, b_t, predicted, p_0, p_1 = self.model(x_0, x_1, l_t_0, l_t_1, h_state, c_state, last=is_last)

                # Store the glimpse location for both frames
                glimpse_location_0.append(l_t_0)
                glimpse_location_1.append(l_t_1)
                
                # Get the glimpse location loss
                log_pi_0.append(p_0)
                log_pi_1.append(p_1)
                
                # Add the baseline
                baselines.append(b_t)

            # Convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi_0 = torch.stack(log_pi_0).transpose(1, 0)
            log_pi_1 = torch.stack(log_pi_1).transpose(1, 0)
            
            # Denormalize the predictions
            predicted_denormalized = torch.stack([denormalize_displacement(l, 100) for l in predicted])
            
            loss_mse = torch.nn.MSELoss()
            loss_l1 = torch.nn.L1Loss()
            
            # Compute the reward based on L1 norm
            R = 1/(1 + torch.sum(torch.abs(predicted_denormalized.detach() - y), dim=1))
            
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)
                
            # Compute losses for differentiable modules
            loss_action = loss_mse(predicted_denormalized, y)
            loss_baseline = loss_mse(baselines, R)

            # Compute reinforce loss, summed over timesteps and averaged across batch
            adjusted_reward = R - baselines.detach()

            loss_reinforce_0 = torch.sum(-log_pi_0 * adjusted_reward, dim=1)
            loss_reinforce_0 = torch.mean(loss_reinforce_0, dim=0) * 0.01
        
            loss_reinforce_1 = torch.sum(-log_pi_1 * adjusted_reward, dim=1)
            loss_reinforce_1 = torch.mean(loss_reinforce_1, dim=0) * 0.01

            # Join the losses
            loss = loss_action + loss_baseline + loss_reinforce_0  + loss_reinforce_1

            # Get the mse
            mse = loss_action
            mae = loss_l1(predicted_denormalized.detach(), y)

            # Store the loss and metric
            losses.update(loss.item(), x_0.size()[0])
            mse_bar.update(mse.item(), x_0.size()[0])
            mae_bar.update(mae.item(), x_0.size()[0])

        return losses.avg, mse_bar.avg, mae_bar.avg

    @torch.no_grad()
    def test(self):
        """Test the RAM model.

        This function should only be called at the very
        end once the model has finished training.
        """
        mse_all = []
        mae_all = []
        samples = []
        
        print(f"[*] Test on {self.num_test} samples")
        
        loss_mse = torch.nn.MSELoss()
        loss_l1 = torch.nn.L1Loss()

        for i, (x, y) in enumerate(self.test_loader):

            # Set data to the respected device
            x_0, x_1, y = x[0].to(self.device), x[1].to(self.device), y.to(self.device)

            # initialize location vector and hidden state
            self.batch_size = x_0.shape[0]
            
            # Reset the model parameters
            h_state, c_state, l_t_0, l_t_1 = self._reset()

            predicted = None

            # For each glimpse
            for t in range(self.num_glimpses):
                
                # Get the prediction on the last glimpse
                is_last = t==self.num_glimpses-1
                
                # Call the model
                h_state, l_t_0, l_t_1, b_t, predicted, p_0, p_1 = self.model(x_0, x_1, l_t_0, l_t_1, h_state, c_state, last=is_last)

            # Denormalize the predictions
            predicted_denormalized = torch.stack([denormalize_displacement(l, 100) for l in predicted])
            
            # Save the first prediction for the each batch
            samples.append([x_0[0], x_1[0], y[0].data, predicted_denormalized[0].data])
            
            # Compute the losses
            mse = loss_mse(predicted_denormalized.detach(), y)
            mae = loss_l1(predicted_denormalized.detach(), y)
            
            mse_all.append(mse.data.cpu().numpy())
            mae_all.append(mae.data.cpu().numpy())
       
        mse_all = sum(mse_all)/len(mse_all)
        mae_all = sum(mae_all)/len(mae_all)
        
        print(f"[*] Test MSE: {mse_all} - MAE: {mae_all}")
        
        for e in samples:
            print(f"Predicted: {e[3]} - Ground-truth: {e[2]}")

    def _reset(self):
        
        h_t = torch.zeros(
            self.batch_size,
            self.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        
        h_state = []
        c_state = []
        
        for i in range(1):
                 
            h_state_i, c_state_i = torch.randn(2,
                                               self.batch_size,
                                               self.hidden_size,
                                               requires_grad=True).to(self.device)
        
            #torch.nn.init.xavier_normal_(h_state_i)
            #torch.nn.init.xavier_normal_(c_state_i)
            
            h_state.append(h_state_i)
            c_state.append(c_state_i)
        
        l_t_0 = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t_0.requires_grad = True
        
        l_t_1 = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t_1.requires_grad = True

        return h_t, c_state, l_t_0, l_t_1

    def _save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_checkpoint.tar"
        
        # Set the checkpoint path
        ckpt_path = os.path.join(self.checkpoint_path, filename)
        
        # Save the checkpoint
        torch.save(state, ckpt_path)
        
        # Save the best model
        if is_best:
            filename = self.model_name + "_best_model.tar"
            
            # Copy the checkpoint to the best model
            shutil.copyfile(ckpt_path, os.path.join(self.checkpoint_path, filename))

    def _load_checkpoint(self, best=False):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print(f"[*] Loading model from {self.checkpoint_path}")

        # Define which model to load
        if best:
            filename = self.model_name + "_best_model.tar"
        else:
            filename = self.model_name + "_checkpoint.tar"
            
        # Set the checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_path, filename)
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Load the variables from checkpoint
        self.start_epoch = checkpoint["epoch"]
        self.best_valid_acc = checkpoint["best_valid_acc"]
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optim_state"])

        if best:
            print(f"[*] Loaded {filename} checkpoint @ epoch {self.start_epoch} with best valid mse of {self.best_valid_acc}")
        else:
            print(f"[*] Loaded {filename} checkpoint @ epoch {self.start_epoch}")
            
            
    def _save_config(self):
            
        df = pd.DataFrame()
        df['patch size'] = [self.patch_size]
        df['glimpse scale'] = [self.glimpse_scale]
        df['num patches'] = [self.num_patches]
        df['num glimpses'] = [self.num_glimpses]
        df['batch size'] = [self.batch_size]
        df['epochs'] = [self.epochs]
        df['lr'] = [self.lr]
        df['num train'] = [self.num_train]
        df['num valid'] = [self.num_valid]
        df['num test'] = [self.num_test]
        
        render_table(df, self.output_path, 'config.jpg')
        
    def _save_results(self):
    
        df = pd.DataFrame()
        df['date'] = ['2016-04-01', '2016-04-02', '2016-04-03']
        df['calories'] = [2200, 2100, 1500]
        df['sleep hours'] = [2200, 2100, 1500]
        df['gym'] = [True, False, False]
        
        # Save the table
        render_table(df, self.output_path, 'results.jpg')
        

def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--test", type=str, required=False, help="should train or test")
    arg.add_argument("--resume", type=str, required=False, help="should resume the train")
    
    args = vars(arg.parse_args())
    
    return args["test"], args["resume"]

if __name__ == "__main__":
    
    args = parse_arguments()

    main = Main(*args)
    
