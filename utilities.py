import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio,structural_similarity

class vae_loss(nn.Module):
    """
    Loss for the Variational AutoEncoder
    
    Params:
        - recon_x : PyTorch tensor
            Reconstructed batch
        - x : PyTorch tensor
            Original batch
        - params : dictionary
            Set of training parameters
        - mu : PyTorch tensor
            Sampled mean from latent space
        - logvar : PyTorch tensor
            Sampled log-variance from latent space
            
    Outputs:
        - loss : PyTorch Tensor of float
            Computed loss
    
    """

    def __init__(self,params):
        super().__init__()
        self.recon_loss = nn.SmoothL1Loss(reduction=params["loss_reduction"])
        self.beta = params["var_beta"]
        
    def forward(self,recon_x,x,mu,logvar):
        rec_loss = self.recon_loss(recon_x,x)
    
        # KL-divergence between the prior distribution over latent vectors
        # (the one we are going to sample from when generating new images)
        # and the distribution estimated by the generator for the given image.
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
        return rec_loss + self.beta * kldivergence


def metrics(recon_batch,batch):
    """
    Evaluate validation set with MAE, SSIM, PSNR
    
    Params:
        - recon_batch : PyTorch tensor
            Reconstructed batch
        - batch : PyTorch tensor
            Original batch
    
    Outputs:
        - maes : list of float
            Set of Mean Absolute Errors for all validation batches
        - ssims : list of float
            Set of Structural Similarity Indices for all validation batches
        - psnrs : list of float
            Set of Peak Signal-to-Noise ratios for all validation batches
    
    
    """
    
    recon_batch = torch.squeeze(recon_batch).cpu().numpy()
    batch = torch.squeeze(batch).cpu().numpy()
    
    maes = []
    ssims = []
    psnrs = []
    
    for i in range(recon_batch.shape[0]):
        norm_recon_image = (recon_batch[i] - recon_batch[i].min())/(recon_batch[i].max() - recon_batch[i].min() + np.finfo(float).eps)
        norm_image = (batch[i] - batch[i].min())/(batch[i].max() - batch[i].min() + np.finfo(float).eps)
        mae = np.sum(np.abs(recon_batch[i].flatten() - batch[i].flatten()))/(np.prod(batch.shape) + np.finfo(float).eps)
        ssim = structural_similarity(np.squeeze(norm_recon_image).astype(float),np.squeeze(norm_image).astype(float))
        psnr = peak_signal_noise_ratio(np.squeeze(norm_recon_image).astype(float),np.squeeze(norm_image).astype(float))
        maes.append(mae)
        ssims.append(ssim)
        psnrs.append(psnr)
        

    return maes,ssims,psnrs


def model_loading(model, optimizer, filename):    
    """
    Load some model that has been saved AFTER TRAINING (NOT A CHECKPOINT)
    
    Usually used for inference
    
    Params:
        - model : PyTorch model 
            Architecture from where checkpoint has been saved
        
        - optimizer : PyTorch optimizer                
            Training optimizer state
        
        - filename : str
            .tar file name where model has been saved
        
    Returns:     
        - model : PyTorch model
            Loaded PyTorch model
            
        - optimizer : PyTorch optimizer
            Loaded PyTorch optimizer
    
    
    """
    
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.

    if os.path.exists(filename):
        print("=> loading model '{}'\n".format(filename))
        checkpoint = torch.load(filename)            
        model.load_state_dict(checkpoint['state_dict'])            
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        try:
            best_ssim = checkpoint["best_ssim"]
        except:
            best_ssim = 0
            
        return model, optimizer,best_ssim
            
    else:        
        print('Non-existing path. Please provide a valid path')
  

def printCurves(metrics,params,log_folder,k=0):
    """
    Print curves for training (and validation, if available)
    
    Params:
        - metrics : dictionary
            Set of computed metrics
        - params : dictionary
            Set of training + validation parameters
        - log_folder : str
            Folder where to store plots with results
        - k : int
            Current fold where data is extracted from (default: 0)

    Outputs:
        Saved plots in folder given

    """
    outfolder = os.path.join(log_folder,"curve_plots","fold{}".format(k))
    if not(os.path.exists(outfolder)):
        os.makedirs(outfolder)
    
    if params["K"] > 1: # Training + validation
        # Loss plot
        fig = plt.figure()
        plt.plot(metrics["iter"],metrics["train_loss"],c="blue",label="Training VAE loss")
        plt.fill_between(metrics["iter"],np.array(metrics["train_loss"]) - np.array(metrics["train_loss_std"]),
                         np.array(metrics["train_loss"]) + np.array(metrics["train_loss_std"]),color="blue",alpha=0.1)
        plt.plot(metrics["iter"],metrics["val_loss"],c="red",label="Validation VAE loss")
        plt.fill_between(metrics["iter"],np.array(metrics["val_loss"]) - np.array(metrics["val_loss_std"]),
                         np.array(metrics["val_loss"]) + np.array(metrics["val_loss_std"]),color="red",alpha=0.1)
        plt.xlabel("Epoch #")
        plt.ylabel("VAE loss")
        plt.legend()
        fig.savefig(os.path.join(outfolder,"loss_plot_fold{}.png".format(k)))
        
        # MAE plot
        fig = plt.figure()
        plt.plot(metrics["iter"],metrics["val_mae"],c="blue",label="Validation MAE")
        plt.fill_between(metrics["iter"],np.array(metrics["val_mae"]) - np.array(metrics["val_mae_std"]),
                         np.array(metrics["val_mae"]) + np.array(metrics["val_mae_std"]),color="blue",alpha=0.1)
        plt.xlabel("Epoch #")
        plt.ylabel("MAE")
        fig.savefig(os.path.join(outfolder,"mae_plot_fold{}.png".format(k)))
        
        # SSIM plot
        fig = plt.figure()
        plt.plot(metrics["iter"],metrics["val_ssim"],c="blue",label="Validation MAE")
        plt.fill_between(metrics["iter"],np.array(metrics["val_ssim"]) - np.array(metrics["val_ssim_std"]),
                         np.array(metrics["val_ssim"]) + np.array(metrics["val_ssim_std"]),color="blue",alpha=0.1)
        plt.xlabel("Epoch #")
        plt.ylabel("SSIM")
        fig.savefig(os.path.join(outfolder,"ssim_plot_fold{}.png".format(k)))
        
        
        # PSNR plot
        fig = plt.figure()
        plt.plot(metrics["iter"],metrics["val_psnr"],c="blue",label="Validation PSNR")
        plt.fill_between(metrics["iter"],np.array(metrics["val_psnr"]) - np.array(metrics["val_psnr_std"]),
                         np.array(metrics["val_psnr"]) + np.array(metrics["val_psnr_std"]),color="blue",alpha=0.1)
        plt.xlabel("Epoch #")
        plt.ylabel("PSNR")
        fig.savefig(os.path.join(outfolder,"psnr_plot_fold{}.png".format(k)))
        
        
    else: # Only training
        # Loss plot
        fig = plt.figure()
        plt.plot(metrics["iter"],metrics["train_loss"],c="blue",label="Training VAE loss")
        plt.fill_between(metrics["iter"],np.array(metrics["train_loss"]) - np.array(metrics["train_loss_std"]),
                         np.array(metrics["train_loss"]) + np.array(metrics["train_loss_std"]),color="blue",alpha=0.1)
        fig.savefig(os.path.join(outfolder,"loss_plot_fold{}.png".format(k)))
        