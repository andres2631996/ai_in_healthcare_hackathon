from dataLoader import Images,augmentation,preprocessing
from sklearn.model_selection import KFold
from model_vgg11_shortSkipPretrained import VAE
import os
import numpy as np
import torch
from torch.utils.data import DataLoader 
import train
import itertools
import time

def getDataLoader(data,params):
    """
    Extract data loader
    
    Params:
        - data : PyTorch dataset
            Dataset to be used for data-loading
        - params : dictionary
            Model and training parameters
            
    Outputs:
        - loader : PyTorch dataloader
            Extracted dataloader
    
    """
    
    loader = DataLoader(data,
                        batch_size=params["batch_size"],
                        num_workers = 0,
                        worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2**32 + id),
                        shuffle=True,
                        pin_memory=True)
    
    return loader



def splitTrainValData(orig_data,train_idx,val_idx):
    """
    Split original dataset into training + validation datasets
    
    Params:
        - orig_data : dataset
            Original dataset
        - train_idx : np.ndarray
            Training indices for original dataset
        - val_idx : np.ndarray
            Validation indices for original dataset
            
    Outputs:
        - trainData : dataset
            Training dataset
        - valData : dataset
            Validation dataset
    
    """
    
    trainData = Images()
    valData = Images()
    
    d = orig_data.dcms
    a = orig_data.ages
    c = orig_data.contrasts
    t = orig_data.tags
    tr = orig_data.trans
    
    trainData.__dcm = d[train_idx]
    trainData.__age = a[:,train_idx]
    trainData.__contrast = c[:,train_idx]
    trainData.__contrastTag = t[:,train_idx]
    trainData.__tran = tr
    
    valData.__dcm = d[val_idx]
    valData.__age = a[:,val_idx]
    valData.__contrast = c[:,val_idx]
    valData.__tag = t[:,val_idx]
    valData.__tran = None
    
    return trainData,valData



# Global variables
    
init = time.time()

path = r"C:\Users\andre\Downloads\kaggle_data"

preprocessing_params = { "resample"        : True,
                         "resolution"      : 1.2,
                         "output_dim"      : 256,
                         "resample_order"  : 2,
                         "equalize"        : True,
                         "clahe_win"       : 2,
                         "clahe_clip_size" : 5,
                         "normalize"       : True}

augmentation_params = { "rot"             : 60,
                        "contrast"        : 0.02, 
                        "bright"          : 0.02,
                        "sigma"           : 2,
                        "points"          : 8,
                        "resample_order"  : 2}

# Model parameters
model_params = {"filters"          : 32,
                "layers"           : 5,
                "batch_norm"       : True,
                "batch_size"       : 1,
                "output_dim"       : preprocessing_params["output_dim"],
                "latent_dims"      : 10,
                "use_gpu"          : True,
                "epochs"           : 200,
                "print_epochs"     : 5,
                "batch_size"       : 1,
                "optimizer"        : "Adam",
                "lr"               : 1e-4,
                "weight_decay"     : 1e-5,
                "lr_scheduling"    : "step",
                "lr_gamma"         : 0.1,
                "step"             : 40000, 
                "plateau_mode"     : "min", 
                "plateau_patience" : 10,
                "var_beta"         : 1,
                "loss_reduction"   : "mean",
                "K" : 5} 

#log = "/content/drive/MyDrive/AI_in_Healthcare_Hackaton/Scripts/Code_Andres/training_code" # Insert some valid direction for TXT file
log = r"C:/Users/andre/Downloads/Full_dataset_hackathon/results/log_pretrainedVAE.txt"
log_folder = os.path.dirname(log)

aug = augmentation(augmentation_params) # Augmentation class
preproc = preprocessing(preprocessing_params) # Preprocessing class

# General arguments for data loading
args = [os.path.join(path,"overview.csv"),
        os.path.join(path,"dicom_dir"),
        preproc,
        True,
        aug]

trainValData, testData = Images(*args).splitTrainTest(0.85)

# Get architecture
net = VAE(model_params,1,1,True)

# Get cross-validation folds
if model_params["K"] > 1:
    kfold = KFold(n_splits=model_params["K"], random_state=2)
    
    cont_fold = 1
    
    train_losses = []
    val_losses = []
    maes = []
    ssims = []
    psnrs = []
    
    for train_idx,val_idx in kfold.split(trainValData):
        
        # Get train and validation datasets and dataloaders
        #trainData,valData = splitTrainValData(trainValData,train_idx,val_idx)
        
        trainData,valData = trainValData.splitTrainTest((train_idx,val_idx))

        loader_train = getDataLoader(trainData,model_params)
        loader_val = getDataLoader(valData,model_params)

        # Training + validation
        print("\nStart training network, fold {}...\n".format(cont_fold))
        
        train_loss,val_loss,mae,ssim,psnr = train.train(net,loader_train,
                                                       loader_val,model_params,
                                                       log,cont_fold)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        maes.append(mae)
        ssims.append(ssim)
        psnrs.append(psnr)
        
        cont_fold += 1
    
    train_losses = np.array(list(itertools.chain.from_iterable(train_losses))).astype(float)
    val_losses = np.array(list(itertools.chain.from_iterable(val_losses))).astype(float)
    maes = np.array(list(itertools.chain.from_iterable(maes))).astype(float)
    ssims = np.array(list(itertools.chain.from_iterable(ssims))).astype(float)
    psnrs = np.array(list(itertools.chain.from_iterable(psnrs))).astype(float)
    
    print("FINAL EVALUATION:")
    print("Train loss: {}+-{}".format(round(train_losses.mean(),4),round(train_losses.std(),4)))
    print("MAE: {}+-{}".format(round(maes.mean(),4),round(maes.std(),4)))
    print("SSIM: {}+-{}".format(round(ssims.mean(),4),round(ssims.std(),4)))
    print("PSNR: {}+-{}".format(round(psnrs.mean(),4),round(psnrs.std(),4)))
    
    if os.path.exists(log):
        with open(log,"a") as f:
            f.write("\nFINAL EVALUATION:\n")
            f.write("Train loss: {}+-{}\n".format(round(train_losses.mean(),4),round(train_losses.std(),4)))
            f.write("MAE: {}+-{}\n".format(round(maes.mean(),4),round(maes.std(),4)))
            f.write("SSIM: {}+-{}\n".format(round(ssims.mean(),4),round(ssims.std(),4)))
            f.write("PSNR: {}+-{}\n".format(round(psnrs.mean(),4),round(psnrs.std(),4)))
        
else:
    loader_train = getDataLoader(trainValData,model_params)
    
    # Training
    print("\nStart training network...\n")

    train_loss,out_model,optimizer,scheduler,loss = train.train(net,loader_train,None,
                                                    model_params,log)
    
    # Model saving
    state = {'iteration'  : model_params["epochs"], 
             'state_dict' : out_model.state_dict(),
             'optimizer'  : optimizer.state_dict(), 
             'lr_sched'   : scheduler,
             'loss'       : loss}

    filename = "pretrainedVAE_epochs{}_lr{}_noCrossVal.tar".format(model_params["epochs"],
                                                                   model_params["lr"])
    
    torch.save(state, os.path.join(log_folder,filename))
    
    
print("Processing done! Time ellapsed: {}h".print(round((time.time()-init)/3600,2)))