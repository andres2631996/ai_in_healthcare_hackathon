import os
import sys
import numpy as np
import time
import torch
import torch.optim as optim
from copy import copy
import utilities
import matplotlib.pyplot as plt
import evaluate


def optimizerExtractor(net,params):    
    """
    Return specified optimizer and if existing, learning rate scheduler.
    
    Params:
        - net : PyTorch model  
            Network where optimizer is applied
        - params : dictionary
            Set of model and training parameters
            
    Returns:
        - optimizer : PyTorch optimizer
            Optimizer for training (Adam/SGD/RMSProp)
        - scheduler : PyTorch learning rate scheduler
            Learning rate scheduler for training (if not False, step/exponential/plateau, else None)
    
    """

    # Possible optimizers. Betas should not have to be changed
    if params["optimizer"] == 'Adam':    
        optimizer = optim.Adam(net.parameters(), 
                               params["lr"], 
                               weight_decay=params["weight_decay"])           
    elif params["optimizer"] == 'RMSprop':        
        optimizer = optim.RMSprop(net.parameters(), 
                                  params["lr"], 
                                  weight_decay=params["weight_decay"])         
    elif params["optimizer"] == 'SGD':        
        optimizer = optim.SGD(net.parameters(), 
                              params["lr"], 
                              weight_decay=params["weight_decay"])          
    else:        
        print('\nWrong optimizer. Please define a valid optimizer (Adam/RMSprop/SGD)\n')        
        sys.exit()

        
    if params["lr_scheduling"] != False:    
        if params["lr_scheduling"] == 'step':        
            scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                                  step_size=params["step"], 
                                                  gamma=params["lr_gamma"])    
        elif params["lr_scheduling"] == 'exponential':        
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 
                                                         gamma=params["lr_gamma"])  
        elif params["lr_scheduling"] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=params["plateau_mode"], 
                                                             factor=params["lr_gamma"], 
                                                             patience=params["plateau_patience"])
        return optimizer, scheduler
    
    else:        
        return optimizer, None



def train(net,loader_train,loader_val,params,log,k=0):
    """
    Main training function
    
    Params:
        - net : PyTorch model
            Architecture to be trained
        - loader_train : PyTorch dataloader
            Training dataloader
        - loader_val : PyTorch dataloader
            Training dataloader
        - params : dictionary
            Set of model and training parameters
        - log : str
            TXT file where to print the training evolution
        - k : int
            Cross-validation fold number during training (default=0)
    
    Returns:
        If cross-validation is applied:
            - all_metrics : dictionary
                Set of mean and std metrics computed (train loss, val loss, val MAE, val SSIM, val PSNR)
            - losses : np.ndarray
                Set of training losses for last training iterations
            - val_losses : list of float
                Set of validation losses for last validation set evaluation (if K == 1, None)
            - maes : list of float
                Set of MAEs for last validation set evaluation (if K == 1, None)
            - ssims : list of float
                Set of SSIMs for last validation set evaluation (if K == 1, None)
            - psnrs : list of float
                Set of PSNRs for last validation set evaluation (if K == 1, None)
                
        Else (no intermediate validation):
            - all_metrics : dictionary
                Set of mean and std metrics computed (train loss)
            - losses : np.ndarray
                Set of training losses for last training iterations
            - net : PyTorch model
                Trained network
            - optimizer : PyTorch optimizer
                Optimizer for saving model
            - scheduler : PyTorch learning rate scheduler
                Learning rate scheduler
            - loss : PyTorch loss function
                Loss function for saving model
        
    
    """
    # Move model to device
    device = torch.device("cuda:0" if params["use_gpu"] and torch.cuda.is_available() else "cpu")
    net.to(device)
    
    # Define loss function
    loss_fun = utilities.vae_loss(params)
    
    ssim_save = 0 # Mean absolute error established as baseline for model saving
    
    net.train()
    optimizer,scheduler = optimizerExtractor(net,params)
    
    # Corresponding model filename
    if loader_val is None:
        filename = "pretrainedVAE_epochs{}_lr{}_noCrossVal.tar".format(params["epochs"],params["lr"])                                                                       
    else:
        filename = "pretrainedVAE_epochs{}_lr{}_fold{}.tar".format(params["epochs"],params["lr"],k)
    
    model_filename = os.path.join(os.path.dirname(log),filename)
        
    if os.path.exists(model_filename):
        # Load pre-trained models
        net,optimizer,ssim_save = utilities.model_loading(net,optimizer,model_filename)
    
    written = 0 # If messages are written or not in log file   
    
    losses = [] # Store losses for each print_epochs interval
    
    # Lists where to store values for curve plotting
    all_train_losses = []
    all_train_losses_std = []
    all_val_losses = []
    all_val_losses_std = []
    
    all_val_ssim = []
    all_val_ssim_std = []
    all_val_psnr = []
    all_val_psnr_std = []
    all_val_mae = []
    all_val_mae_std = []
    
    all_iter = []
    
    for i in range(params["epochs"]):

        num_batches = 0
        batch_accumulator = 0
        
        for batch in loader_train:
            # Feed batch to model
            recon_batch,mu_batch,logvar_batch,latent_var = net(batch,device)
            
            # loss
            loss = loss_fun(recon_batch,batch["DCM"].to(device),mu_batch,logvar_batch)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()
            
            if params["lr_scheduling"] != False:
                scheduler.step()
            
            batch_accumulator += loss.item()
            num_batches += 1
            
        losses.append(batch_accumulator/(num_batches + np.finfo(float).eps))
            
        if (i%params["print_epochs"] == 0) or (i == params["epochs"]-1):
            losses = np.array(losses).astype(float)
            mean_loss = losses.mean()/params["batch_size"]
            std_loss = losses.std()/params["batch_size"]
            
            print("Training: Fold: {}, Iteration: {} // Loss distribution: {}+-{}".format(k,i,round(mean_loss,4),round(std_loss,4)))
            
            if log != "":
                log_folder = os.path.dirname(log)
                if not(os.path.exists(log_folder)):
                    os.makedirs(log_folder)
            else:
                log_folder = None
            
            # Validation set evaluation
            if params["K"] > 1:
                all_iter.append(i)
                all_train_losses.append(mean_loss)
                all_train_losses_std.append(std_loss)
            
                val_losses,maes,ssims,psnrs = evaluate.evaluate(net,loader_val,params,k,i,log_folder)
                
                mean_val_loss = np.array(val_losses).mean()/params["batch_size"]
                std_val_loss = np.array(val_losses).std()/params["batch_size"]
                mean_val_mae = np.array(maes).mean()/params["batch_size"]
                std_val_mae = np.array(maes).std()/params["batch_size"]
                mean_val_ssim = np.array(ssims).mean()/params["batch_size"]
                std_val_ssim = np.array(ssims).std()/params["batch_size"]
                mean_val_psnr = np.array(psnrs).mean()/params["batch_size"]
                std_val_psnr = np.array(psnrs).std()/params["batch_size"]
                
                all_val_losses.append(mean_val_loss)
                all_val_losses_std.append(std_val_loss)
                
                all_val_mae.append(mean_val_mae)
                all_val_mae_std.append(std_val_mae)
                
                all_val_ssim.append(mean_val_ssim)
                all_val_ssim_std.append(std_val_ssim)
                
                all_val_psnr.append(mean_val_psnr)
                all_val_psnr_std.append(std_val_psnr)
                
                print("Validation: Fold: {}, Iteration: {} // Loss: {}+-{} // MAE: {}+-{} // SSIM: {}+-{} // PSNR: {}+-{}".format(k,i,round(mean_val_loss,4),round(std_val_loss,4),round(mean_val_mae,4),round(std_val_mae,4),round(mean_val_ssim,4),round(std_val_ssim,4),round(mean_val_psnr,4),round(std_val_psnr,4)))
                
                if ssim_save < mean_val_ssim:
                    state = {'iteration'  : i + 1, 
                             'state_dict' : net.state_dict(),
                             'optimizer'  : optimizer.state_dict(),
                             'lr_sched'   : scheduler,
                             'loss'       : loss, 
                             'best_ssim'  : mean_val_ssim}

                    filename = "pretrainedVAE_epochs{}_lr{}_fold{}.tar".format(params["epochs"],
                                                                                params["lr"],k)
    
                    torch.save(state, os.path.join(log_folder,filename))
    
                    print('Validation SSIM has improved from {} to {}. Saved model\n'.format(ssim_save,mean_val_ssim))
                    ssim_save = copy(mean_val_ssim)
            else:
                all_iter.append(i)
                all_train_losses.append(mean_loss)
                all_train_losses_std.append(std_loss)

            if not(os.path.exists(log)) or i == 0:
                flag = "w"
            else:
                flag = "a"
                
            if log != "":
                with open(log,flag) as f:
                    if i == 0 and written == 0:
                        f.write("\nCROSS-VALIDATION: FOLD {}\n".format(i)) 
                        written = 1
                    f.write("Training: Iteration: {} // Loss: {}+-{}\n".format(i,round(mean_loss,4),round(std_loss,4)))
                    if params["K"] > 1:
                        f.write("Validation: Iteration: {} // Loss: {}+-{} // MAE: {}+-{} // SSIM: {}+-{} // PSNR: {}+-{}\n".format(i,round(mean_val_loss,4),round(std_val_loss,4),round(mean_val_mae,4),round(std_val_mae,4),round(mean_val_ssim,4),round(std_val_ssim,4),round(mean_val_psnr,4),round(std_val_psnr,4)))
                    if i == params["epochs"]-1:
                        f.write("\n---------------------------------------\n")
                    f.close()
                    
            losses = []
        
    if params["K"] > 1:
        all_metrics = {"train_loss"     : all_train_losses,
                       "train_loss_std" : all_train_losses_std,
                       "val_loss"       : all_val_losses,
                       "val_loss_std"   : all_val_losses_std,
                       "val_mae"        : all_val_mae,
                       "val_mae_std"    : all_val_mae_std,
                       "val_ssim"       : all_val_losses,
                       "val_ssim_std"   : all_val_ssim_std,
                       "val_psnr"       : all_val_psnr,
                       "val_psnr_std"   : all_val_psnr_std,
                       "iter"           : all_iter}

        utilities.printCurves(all_metrics,params,log_folder,k)

        return losses,val_losses,maes,ssims,psnrs
    else:
        all_metrics = {"train_loss"     : all_train_losses,
                       "train_loss_std" : all_train_losses_std,
                       "iter"           : all_iter} 
        
        utilities.printCurves(all_metrics,params,log_folder,k)
        
        return losses,net,optimizer,scheduler,loss