import os
import sys
import subprocess
import numpy as np
import math
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools
import random
import utilities



def evaluate(net,loader,params,k,it,log):
    """
    Evaluate model with validation set
    
    Params:
        - net : PyTorch model
            Network to be validated
        - loader : PyTorch dataloader
            Validation dataloader
        - params : dictionary
            Set of model parameters
        - k : int
            Fold number for saving information in log folder
        - it : int
            Iteration #
        - log : str
            Folder where to save example validation images
    
    Outputs:
        - losses : list of float
            Set of VAE losses for all validation batches
        - maes : list of float
            Set of Mean Absolute Errors for all validation batches
        - ssims : list of float
            Set of Structural Similarity Indices for all validation batches
        - psnrs : list of float
            Set of Peak Signal-to-Noise ratios for all validation batches
    
    """
    loss_fun = utilities.vae_loss(params)
    device = torch.device("cuda:0" if params["use_gpu"] and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        net.eval()
        losses = []
        maes = []
        ssims = []
        psnrs = []
        
        ind_plot = random.randint(0,len(loader)-1)
        
        for enum_loader,batch in enumerate(loader):
            
            # Feed batch to model
            recon_batch,mu_batch,logvar_batch,latent_var = net(batch,device)
            
            batch = batch["DCM"].to(device)
            
            # Loss
            loss = loss_fun(recon_batch, batch, mu_batch, logvar_batch)
            losses.append(loss.item())
            
            # SSIM and PSNR metrics
            mae,ssim,psnr = utilities.metrics(recon_batch,batch)
            maes.append(mae)
            ssims.append(ssim)
            psnrs.append(psnr)
            
            # Save reconstruction example in log folder
            if enum_loader == ind_plot and log is not None and os.path.exists(log):
                fig = plt.figure()
                plt.subplot(121)
                plt.imshow(torch.squeeze(batch[0]).cpu(),cmap="gray")
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(torch.squeeze(recon_batch[0]).cpu(),cmap="gray")
                plt.colorbar()
                fig.savefig(os.path.join(log,"output_comparison_pretrainedVAE_fold{}_it{}.png".format(k,it)))
            
        maes = list(itertools.chain.from_iterable(maes))
        ssims = list(itertools.chain.from_iterable(ssims))
        psnrs = list(itertools.chain.from_iterable(psnrs))
        
        return losses,maes,ssims,psnrs