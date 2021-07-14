import os
import time
import math
import shutil
import pickle
import argparse
import cv2
import csv
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from tqdm import tqdm
from ramvo import RAMVO
from utils import *
from shutil import copyfile
from data_loader import get_data_loader
from torchviz import make_dot
torch.set_printoptions(threshold=10_000)
torch.set_printoptions(profile="full", precision=10, linewidth=100, sci_mode=False)
#torch.autograd.set_detect_anomaly(True)

# The percentage error formula
percentage_error_formula = lambda x, amount_variation: round(x/amount_variation*100, 3)       

def mse(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.subtract(actual,pred)
    #return np.square(np.subtract(actual,pred)).mean()

class Main:
    
    def __init__(self):
        
        # Glimpse Network Params
        self.num_glimpses = 8 # number of glimpses, i.e. BPTT iterations
        self.patch_size = 32 # size of extracted patch at highest res
        self.num_patches = 3 # number of downscaled patches per glimpse
        self.glimpse_scale = 3 # scale of successive patches
        
        # Data Params
        self.batch_size = 128 # number of images in each batch of data
        self.num_workers = 4 # number of subprocesses to use for data loading
        self.num_channels = 1
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.num_train = None
        self.num_valid = None
        self.num_test = None
            
        # Training params
        self.epochs = 400 # number of epochs to train for
        self.start_epoch = 0
        self.momentum = 0.5 # Nesterov momentum value
        self.lr = 1e-4 # Initial learning rate value
        self.lr_patience = 150 # Number of epochs to wait before reducing lr
        self.lr_threshold = 0.01
        self.train_patience = 25 # Number of epochs to wait before stopping train

        # Other params
        self.random_seed = 3 # Seed to ensure reproducibility
        self.best = True # Load best model or most recent for testing
        self.print_freq = 10 # How frequently to print training details
        self.pin_memory = False
        self.preload = False
        self.best_valid_mae = 10000000.0
        self.counter = 0
        self.elapsed_time = 0
        self.num_parameters = 0
        
        # Set the seed
        torch.manual_seed(self.random_seed)
        
         # Check if the gpu is available
        if torch.cuda.is_available():
            
            self.device = torch.device("cuda")
            
            torch.cuda.manual_seed(self.random_seed)
            
            self.num_workers = 1
            self.pin_memory = True
            self.preload = True
            
        else:
            self.device = torch.device("cpu")

        print(f"\n[*] Device: {self.device}")
        
        # Build the model
        self.model = RAMVO(
            self.batch_size,
            self.patch_size,
            self.num_patches,
            self.num_glimpses,
            self.glimpse_scale,
            self.num_channels,
            self.device
        )
        
        # Set the model to the device
        self.model.to(self.device)
        
        # Start the optimizer
        self.optimizer = torch.optim.Adam([
                {'params': self.model.glimpse.parameters()},
                {'params': self.model.core.parameters()},
                {'params': self.model.regressor.parameters()},
                {'params': self.model.locator.parameters(), 'lr': 1e-5},
                {'params': self.model.baseliner.parameters(), 'lr': 1e-4},
            ], lr=self.lr)
        
        # Start the scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=self.lr_patience, threshold=self.lr_threshold)           

        # Count the number of the model parameters
        self._count_parameters()

        
    def _load_dataset(self, dataset, batch_size, train_seq=[0, 2, 4, 5, 6, 8, 9], val_seq=[10], test_seq=3):

        # Set the data loader
        self.train_loader, self.valid_loader, self.test_loader = get_data_loader(
            batch_size,
            dataset,
            train_seq,
            val_seq,
            test_seq,
            self.num_workers,
            self.pin_memory,
            self.preload,
            self.random_seed
        )
        
        if train_seq is not None:
            self.num_train = len(self.train_loader.sampler)
        if val_seq is not None:
            self.num_valid = len(self.valid_loader.sampler)
        if test_seq is not None:
            self.num_test = len(self.test_loader.sampler)
            
    def train(self, resume=None, plot_graph=True):

        self.plot_graph = plot_graph
        
        # Load the dataset
        self._load_dataset(dataset='kitti', batch_size=self.batch_size) #dataset='euroc', train_seq=[0]
        
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
            self.heatmap_path = os.path.join(self.output_path, 'heatmap')
            self.loss_path = os.path.join(self.output_path, 'loss')
            
            # Create the folders
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if not os.path.exists(self.glimpse_path):
                os.makedirs(self.glimpse_path)
            if not os.path.exists(self.heatmap_path):
                os.makedirs(self.heatmap_path)
            if not os.path.exists(self.loss_path):
                os.makedirs(self.loss_path)
        
        else:
            
            # Set the model to be loaded
            self.model_name = resume
            
            # Set the folders
            self.output_path = os.path.join('out', self.model_name)
            self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
            self.glimpse_path = os.path.join(self.output_path, 'glimpse')
            self.heatmap_path = os.path.join(self.output_path, 'heatmap')
            self.loss_path = os.path.join(self.output_path, 'loss')
            
            # Load the model
            self._load_checkpoint(best=False)            
        
        print(f"[*] Output Folder: {self.model_name}")
        print(f"[*] Total Trainable Params: {self.num_parameters}")
        print(f"[*] Train on {self.num_train} samples, validate on {self.num_valid} samples")

        tic = time.time()
        
        # For each epoch
        for epoch in range(self.start_epoch, self.epochs):

            # Get the current lr
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"\nEpoch: {epoch+1}/{self.epochs} - LR: {current_lr}")

            # Train one epoch
            train_mse, train_mse_rot, train_mse_tran, train_rl, train_entropy, train_data = self._train_one_epoch(epoch)

            # Validate one epoch
            val_mse, val_mse_rot, val_mse_tran, val_rl, val_data = self._validate(epoch)

            # Reduce lr if validation loss plateaus
            self.scheduler.step(train_mse)

            # Check if it is the best model
            is_best = val_mse < self.best_valid_mae
            
            msg = "train rot: {:.6f}, train tran: {:.6f}, train RL: {:.6f}, H: {:.3f} | val rot: {:.6f}, val tran: {:.6f}, val RL: {:.6f}"
            
            # Check for improvement
            if is_best:
                self.counter = 0
                msg += " [*]"
            else:
                self.counter += 1
            
            print(msg.format(train_mse_rot, train_mse_tran, train_rl, train_entropy, val_mse_rot, val_mse_tran, val_rl))
            
            self.best_valid_mae = min(val_mse, self.best_valid_mae)
            
            # Save the checkpoint for each epoch
            self._save_checkpoint({
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "sched_state": self.scheduler.state_dict(),
                    "best_valid_mae": self.best_valid_mae,
                }, is_best
            )
            
            # Dump the losses
            with open(os.path.join(self.loss_path, f"loss_epoch_{epoch+1}.p"), "wb") as f:
                
                data = (train_data, val_data)
                
                pickle.dump(data, f)
                
        toc = time.time()
        
        self.elapsed_time = toc - tic
                
        # Save the configuration as image
        self._save_config()       

    def _train_one_epoch(self, epoch):
  
        self.model.train()
        
        # Create the loss object
        loss_mse = torch.nn.MSELoss()
        
        batch_time = AverageMeter()
        mse_bar = AverageMeter()
        mse_rot_bar = AverageMeter()
        mse_tran_bar = AverageMeter()
        reinforce_bar = AverageMeter()
        entropy_bar = AverageMeter()
        
        # Store the losses array
        loss_regressor_array = []
        loss_rot_array = []
        loss_tran_array = []
        reward_array = []
        loss_reinforce_array = []
        loss_baseline_array = []

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            
            # For each minibatch
            for i, (x, op, y) in enumerate(self.train_loader):
                                       
                # Set data to the respected device
                x, op, y = x.to(self.device), op.to(self.device), y.to(self.device)
               
                # Call the model and pass the minibatch
                predicted, l_t_array, log_pi_array, baseline_array, entropy_array = self.model(x, op)

                # Convert list to tensors and reshape
                baselines = baseline_array.transpose(1, 0)
                log_pi = log_pi_array.transpose(1, 0)
                entropy = entropy_array.transpose(1, 0)
            
                # Separate the rot and trans components
                y_rot = y[:, :3]
                y_tran = y[:, 3:]
                pred_rot = predicted[:, :3]
                pred_tran = predicted[:, 3:]
                
                # Define the vo reward function
                vo_reward = torch.square(torch.sub(pred_rot.detach(), y_rot)).mean(dim=1) + torch.square(torch.sub(pred_tran.detach(), y_tran)).mean(dim=1)
                
                R = 1/(1 + vo_reward) 
                
                R_unsqueeze = R.unsqueeze(1).repeat(1, self.num_glimpses)
                     
                # Compute losses for differentiable modules
                loss_rot = loss_mse(pred_rot, y_rot)
                loss_tran = loss_mse(pred_tran, y_tran)
                
                loss_regressor = loss_rot + loss_tran
                
                loss_baseline = loss_mse(baselines, R_unsqueeze)
                
                # Compute reinforce loss, summed over timesteps and averaged across batch
                adjusted_reward = R_unsqueeze - baselines.detach()

                loss_reinforce_sum = torch.sum(-log_pi * adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce_sum, dim=0)
                
                # Join the losses
                loss = loss_regressor + loss_reinforce + loss_baseline
            
                self.optimizer.zero_grad()
                
                # Update the weights
                loss.backward()
                
                #plot_grad_flow(self.model.named_parameters())
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                                
                self.optimizer.step()
                
                # Store the losses
                loss_regressor_array.append(loss_regressor.cpu().data.numpy())
                loss_rot_array.append(loss_rot.cpu().data.numpy())
                loss_tran_array.append(loss_tran.cpu().data.numpy())
                reward_array.append(torch.mean(R).cpu().data.numpy())  
                loss_reinforce_array.append(loss_reinforce_sum.abs().mean().cpu().data.numpy())
                loss_baseline_array.append(loss_baseline.cpu().data.numpy())
                
                # Store the metrics
                mse_bar.update(loss_regressor.item())
                mse_rot_bar.update(loss_rot.item())
                mse_tran_bar.update(loss_tran.item())
                reinforce_bar.update(loss_reinforce_sum.abs().mean().item())
                entropy_bar.update(entropy.mean().item())

                # Measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                # Set the var description
                pbar.set_description(("{:.1f}s - train rot: {:.6f}, train tran: {:.6f}, train RL: {:.6f}".format((toc-tic), loss_rot.item(), loss_tran.item(), loss_reinforce_sum.abs().mean().item())))
                
                # Update the bar
                pbar.update(self.batch_size)
                
                # Plot the graph
                if i == 0 and self.plot_graph:
                       
                    print(f"[*] Plotting the graph")
                    
                    # Generate the plot
                    make_dot(loss, params=dict(self.model.named_parameters()), engine="neato").render(os.path.join(self.output_path, "ramvo_neato"), format="pdf", cleanup=True)
                    make_dot(loss, params=dict(self.model.named_parameters()), engine="dot").render(os.path.join(self.output_path, "ramvo_dot"), format="pdf", cleanup=True)

                # Save glimpses for the heatmap every 5 minibatches
                if i % 5 == 0: #  or epoch == 0
                    
                    #trans = transforms.ToPILImage()
                    
                    # Format the data for storage
                    #img_0 = [g.cpu() for g in x[:, 0]]
                    #img_1 = [g.cpu() for g in x[:, 1]]
                    #loc = [l.cpu().data.numpy() for l in glimpse_location]
                    
                    # Build the data to be saved
                    #data = ((img_0, loc), (img_1, loc))
                    
                    # Dump the glimpses
                    #with open(os.path.join(self.glimpse_path, f"glimpses_epoch_{epoch+1}.p"), "wb") as f:
                    #    pickle.dump(data, f)
                    
                    # Dump the glimpses for heatmap
                    with open(os.path.join(self.heatmap_path, f"epoch_{epoch+1}_minibatch_{i}.p"), "wb") as f:
                        pickle.dump(l_t_array, f)
                
            # Build the train data array
            train_data = (reward_array, loss_regressor_array, loss_rot_array, loss_tran_array, loss_baseline_array, loss_reinforce_array)

            # Convert to numpy array
            train_data = map(np.asarray, train_data)

            return mse_bar.avg, mse_rot_bar.avg, mse_tran_bar.avg, reinforce_bar.avg, entropy_bar.avg, train_data

    @torch.no_grad()
    def _validate(self, epoch):
                   
        loss_mse = torch.nn.MSELoss()
 
        mse_bar = AverageMeter()
        mse_rot_bar = AverageMeter()
        mse_tran_bar = AverageMeter()
        reinforce_bar = AverageMeter()
            
        # Store the losses array
        loss_regressor_array = []
        loss_rot_array = []
        loss_tran_array = []
        reward_array = []
        loss_reinforce_array = []
        loss_baseline_array = []

        for i, (x, op, y) in enumerate(self.valid_loader):
                                   
            # Set data to the respected device
            x, op, y = x.to(self.device), op.to(self.device), y.to(self.device)
           
            # Call the model and pass the minibatch
            predicted, l_t_array, log_pi_array, baseline_array, _ = self.model(x, op)

            # Convert list to tensors and reshape
            baselines = baseline_array.transpose(1, 0)
            log_pi = log_pi_array.transpose(1, 0)
                        
            y_rot = y[:, :3]
            y_tran = y[:, 3:]
            pred_rot = predicted[:, :3]
            pred_tran = predicted[:, 3:]
            
            # Define the vo reward function
            vo_reward = torch.square(torch.sub(pred_rot.detach(), y_rot)).mean(dim=1) + torch.square(torch.sub(pred_tran.detach(), y_tran)).mean(dim=1)

            R = 1/(1 + vo_reward)
            
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)

            # Compute losses for differentiable modules
            loss_rot = loss_mse(pred_rot, y_rot)
            loss_tran = loss_mse(pred_tran, y_tran)
            
            loss_regressor = loss_rot + loss_tran
            
            loss_baseline = loss_mse(baselines, R) 

            # Compute reinforce loss, summed over timesteps and averaged across batch
            adjusted_reward = R - baselines.detach()

            loss_reinforce_sum = torch.sum(-log_pi * adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce_sum, dim=0) 

            # Join the losses
            loss = loss_regressor + loss_reinforce + loss_baseline
            
            # Store the losses
            loss_regressor_array.append(loss_regressor.cpu().data.numpy())
            loss_rot_array.append(loss_rot.cpu().data.numpy())
            loss_tran_array.append(loss_tran.cpu().data.numpy())
            reward_array.append(torch.mean(R).cpu().data.numpy())  
            loss_reinforce_array.append(loss_reinforce_sum.abs().mean().cpu().data.numpy())
            loss_baseline_array.append(loss_baseline.cpu().data.numpy())

            # Store the metrics
            mse_bar.update(loss_regressor.item())
            mse_rot_bar.update(loss_rot.item())
            mse_tran_bar.update(loss_tran.item())
            reinforce_bar.update(loss_reinforce_sum.abs().mean().item())

        # Build the validation data array
        validation_data = (reward_array, loss_regressor_array, loss_rot_array, loss_tran_array, loss_baseline_array, loss_reinforce_array)
            
        # Convert to numpy array
        validation_data = map(np.asarray, validation_data)

        return mse_bar.avg, mse_rot_bar.avg, mse_tran_bar.avg, reinforce_bar.avg, validation_data

    @torch.no_grad()
    def test(self, model_name, dataset, test_seq):
        
        # Set the model to load
        self.model_name = model_name
        
        # Load the dataset
        self._load_dataset(dataset=dataset, batch_size=self.batch_size, train_seq=None, val_seq=None, test_seq=test_seq)
        
        # Set the folders
        self.output_path = os.path.join('out', self.model_name, 'results', str(test_seq))
        self.checkpoint_path = os.path.join('out', self.model_name, 'checkpoint')
               
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
               
        # Copy gt
        if dataset == 'kitti':
            copyfile(f"/mnt/ssd/dataset/kitti/poses/{test_seq:02d}.txt", f"{self.output_path}/groundtruth_kitti.txt")
        else:
            seq_name = move_euroc_gt(test_seq)
            copyfile(f"../dataset/euroc/{seq_name}/state_groundtruth_estimate0/data_processed.csv", f"{self.output_path}/groundtruth_euroc.csv")
        
        # Load the model
        self._load_checkpoint(best=False)
        
        # Get the best policy
        self.model.locator.test = False
        
        # Print the model info
        print(f"[*] Total Trainable Params: {self.num_parameters}")
            
        mse_all = []
        samples = []

        predictions_array = torch.tensor([]).to(self.device)
        y_array = torch.tensor([]).to(self.device)
        l_t_array_all = torch.tensor([]).to(self.device)
        
        print(f"[*] Test on {self.num_test} samples")
        
        loss_mse = torch.nn.MSELoss()    

        for i, (x, op, y) in enumerate(self.test_loader):
                                           
            # Set data to the respected device
            x, op, y = x.to(self.device), op.to(self.device), y.to(self.device)
            
            # Call the model and pass the mini batch
            predicted, l_t_array, _, _, _ = self.model(x, op)
      
            predictions_array = torch.cat((predictions_array, predicted))
            y_array = torch.cat((y_array, y))
            l_t_array_all = torch.cat((l_t_array_all, l_t_array), axis=1)
            
            # For the first minibatch
            if i == 0:
                
                trans = transforms.Compose([
                    NormalizeInverse([0.4205234349], [0.2877691686]),
                    transforms.ToPILImage()
                ])
                                   
                # Build the glimpses array
                glimpses = [trans(x[0, 0].cpu()), trans(x[0, 1].cpu()), l_t_array[:, 0].cpu().data.numpy()]

                # Dump the glimpses
                with open(os.path.join(self.output_path, f"glimpses_epoch_test.p"), "wb") as f:
                    pickle.dump(glimpses, f)
       
        # Dump the glimpses for heatmap
        with open(os.path.join(self.output_path, f"glimpses_heatmap.p"), "wb") as f:
            pickle.dump(l_t_array_all, f)
                         
        # Get samples every 20 frames
        skip = len(y_array) // 20
                         
        # Save the first prediction
        samples = [[h.cpu().numpy(), p.cpu().numpy()] for h, p in zip(y_array[::skip], predictions_array[::skip])]
            
        # Compute the metrics
        y_rot = y_array[:, :3]
        y_tran = y_array[:, 3:]
        pred_rot = predictions_array[:, :3]
        pred_tran = predictions_array[:, 3:]  
        
        # Compute losses for differentiable modules
        rot_loss = loss_mse(pred_rot, y_rot)
        trans_loss = loss_mse(pred_tran, y_tran)
        
        regressor_loss = rot_loss + trans_loss
        
        # Save the results as image
        self._save_results(regressor_loss.item(), rot_loss.item(), trans_loss.item(), samples, glimpses)
               
        mean = torch.tensor([-7.6397992e-05, 2.6872402e-04, 4.7161593e-06, -9.7197731e-04, -1.7675826e-02, 9.2309231e-01]).to(self.device)
        std = torch.tensor([0.00305257, 0.01770405, 0.00267268, 0.02503707, 0.01716818, 0.30884704]).to(self.device)
   
        # Denormalize gt
        std_inv = 1 / (std + 1e-8)
        mean_inv = -mean * std_inv
        
        y_array = (y_array - mean_inv) / std_inv
        predictions_array = (predictions_array - mean_inv) / std_inv
        
        predictions_array = predictions_array.cpu().data.numpy()
        y_array = y_array.cpu().data.numpy()
        
        # Generate the trajectory and metrics
        self._save_evaluation(predictions_array, dataset)
    
    def _examine_batch(self, x_array, y_array, pred_array):
        
        # Compose the transformations
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4561], [0.3082])
        ])
        
        transinv = transforms.Compose([
                NormalizeInverse([0.4561], [0.3082]),
                transforms.ToPILImage()
        ])

        for x, y, p in zip(x_array, y_array, pred_array):
            
            y_rot = y[:3]
            y_tran = y[3:]
            pred_rot = p[:3]
            pred_tran = p[3:]
            
            error = mse(pred_rot, y_rot) + mse(pred_tran, y_tran)
            
            print(error)
            
            first_image = x[0]
            second_image = x[1]

            first_image = transinv(first_image)
            second_image = transinv(second_image)

            first_image = cv2.cvtColor(np.array(first_image), cv2.COLOR_RGB2BGR)
            second_image = cv2.cvtColor(np.array(second_image), cv2.COLOR_RGB2BGR)

            # Show the image
            cv2.imshow("First", first_image)
            cv2.imshow("Second", second_image)
            
            cv2.waitKey()
        
    def _count_parameters(self, print_table=False):
    
        table = PrettyTable(["Modules", "Parameters"])

        for name, parameter in self.model.named_parameters():
            
            if parameter.requires_grad:
                param = parameter.numel()
                table.add_row([name, param])
                
                self.num_parameters += param

        if print_table:
            print(table)

    def _save_checkpoint(self, state, is_best):

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
        self.best_valid_mae = checkpoint["best_valid_mae"]
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optim_state"])
        self.scheduler.load_state_dict(checkpoint["sched_state"])

        if best:
            print(f"[*] Loaded {filename} checkpoint @ epoch {self.start_epoch} with best valid mae of {self.best_valid_mae}")
        else:
            print(f"[*] Loaded {filename} checkpoint @ epoch {self.start_epoch}")
            
            
    def _save_config(self):
            
        df = pd.DataFrame()
        df['patch size'] = [self.patch_size]
        df['glimpse scale'] = [self.glimpse_scale]
        df['num patches'] = [self.num_patches]
        df['num glimpses'] = [self.num_glimpses]
        df['batch size'] = [self.batch_size]
        df['lr'] = [self.lr]
        df = df.astype(str)
        
        # Render the table
        render_table(df, self.output_path, 'config_1.svg')
        
        df = pd.DataFrame()
        df['epochs'] = [self.epochs]
        df['num train'] = [self.num_train]
        df['num valid'] = [self.num_valid]
        df['num test'] = [self.num_test]
        df['# Params'] = [self.num_parameters]
        df['Time'] = [time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))]
        df = df.astype(str)
        
        # Render the table
        render_table(df, self.output_path, 'config_2.svg')
        
    def _save_results(self, mse_all, rot, tran, samples, glimpses):
        
        loss_mse = torch.nn.MSELoss()   
        np.set_printoptions(threshold=np.inf) 
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=6)
    
        df = pd.DataFrame()
        df['Regressor'] = [round(mse_all, 6)]
        df['Rot'] = [round(rot, 6)]
        df['Tran'] = [round(tran, 6)]

        df = df.astype(str)
        
        print(df)
        
        # Save the table
        render_table(df, self.output_path, 'metrics.svg')
        
        predictions_array = []
        ground_truth_array = []
        mse_array = []
        
        for e in samples:
            
            q = torch.tensor(e[0])
            p = torch.tensor(e[1])
            
            # Compute the metrics
            y_rot = q[:3]
            y_tran = q[3:]
            pred_rot = p[:3]
            pred_tran = p[3:]

            # Compute losses for differentiable modules
            mse = loss_mse(pred_rot, y_rot) + loss_mse(pred_tran, y_tran)
            
            predictions_array.append(p.numpy())
            ground_truth_array.append(q.numpy())
            mse_array.append(mse)
        
        df = pd.DataFrame()
        df['Predicted'] = predictions_array
        df['Ground-truth'] = ground_truth_array
        df['MSE'] = list(map(lambda x: "%.6f" % x, mse_array))
        
        df = df.astype(str)
        
        print(df)
        
        # Save the table
        render_table(df, self.output_path, 'predictions.svg', col_width=8)
        
    def _save_evaluation(self, predictions, dataset):
        
        if dataset == 'kitti':
        
            trajectory = []
            
            mat = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            
            for i in predictions:
                
                rot = eulerAnglesToRotationMatrix(i[:3])
                trans = i[3:].reshape(-1, 1)
                
                current = np.concatenate((rot, trans), axis=1)
                current = np.concatenate((current, np.asarray([[0, 0, 0, 1]])))

                mat = np.matmul(mat, current)          

                trajectory.append(mat[:3].flatten())
                
            with open(f"{self.output_path}/prediction.txt", "w") as f:
                writer = csv.writer(f, delimiter =' ')
                writer.writerows(trajectory)
            
        else:
            
            trajectory = []
            
            mat =  np.asarray([[-0.36623078,  0.35140725, -0.86161938,  4.686208  ],
                             [ 0.14862573,  0.93615371,  0.31863244, -1.784735  ],
                             [ 0.91857793, -0.0113658,  -0.39507646,  0.843777  ]])
            
            for i in predictions:
                
                rot = eulerAnglesToRotationMatrix(i[:3])
                trans = i[3:].reshape(-1, 1)
                
                current = np.concatenate((rot, trans), axis=1)
                current = np.concatenate((current, np.asarray([[0, 0, 0, 1]])))

                mat = np.matmul(mat, current)          

                trajectory.append(mat[:3])
                
            
            trajectory_euroc = []
            
            for i, traj in enumerate(trajectory):
                
                quat = rotation_matrix_to_quaternion(traj[:, :3])
                           
                trans = traj[:3, -1]
                
                pose = np.concatenate((np.asarray([i]), trans, quat, np.zeros(9)))
                
                trajectory_euroc.append(np.round(pose.flatten(), 6))
            
            with open(f"{self.output_path}/prediction.txt", "w") as f:
                writer = csv.writer(f, delimiter =',')
                writer.writerows(trajectory_euroc)
            
    
def parse_arguments():
    
    arg = argparse.ArgumentParser()
    arg.add_argument("--test", type=str, required=False, help="should train or test")
    arg.add_argument("--dataset", type=str, required=False, help="test dataset")
    arg.add_argument("--test_seq", type=int, required=False, help="test sequence")
    arg.add_argument("--resume", type=str, required=False, help="should resume the train")
    arg.add_argument("--plot_graph", type=str2bool, required=False, help="should plot the graph")
    
    args = vars(arg.parse_args())
    
    return args


if __name__ == "__main__":
    
    args = parse_arguments()

    main = Main()
    
    if args['test'] is not None:
        main.test(args['test'], args['dataset'], args['test_seq'])
    else:
        main.train(args['resume'], args['plot_graph'])
 
