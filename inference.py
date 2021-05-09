from dataLoader import Images,augmentation,preprocessing
from model_vgg11_shortSkipPretrained import VAE
import os
import numpy as np
import torch
from torch.utils.data import DataLoader 
import train
import matplotlib.pyplot as plt
import getopt
import utilities
import time
import sys

def main(argv):
    """
    Perform inference on lung CT slices dataset
    
    Params:
        - i : input test data path
        - m : saved model filename
        - o : output path for inference results
        
    Outputs:
        Saved inference results on the given data, with the given model, in the given output path
    
    """
    
    init = time.time()
    i = ""
    m = ""
    o = ""
    
    try:
        opts,args = getopt.getopt(argv,"hi:m:o:",["in=","model=","out="])
    except getopt.GetoptError:
        print("inference.py -i <in_folder> -m <model_file__tar> -o <out_folder>")
        sys.exit()
        
    for opt,arg in opts:
        if opt == '-h':
            print("inference.py -i <in_folder> -m <model_file__tar> -o <out_folder>")
            sys.exit()
        elif opt in ("-i","in"):
            i = arg
        elif opt in ("-m","model"):
            m = arg
        elif opt in ("-o","out"):
            o = arg
            
    if i == "" or m == "":
        print("inference.py -i <in_folder> -m <model_file__tar> -o <out_folder>")
        sys.exit()
        
    if not(os.path.exists(i)) or not(os.path.isdir(i)):
        print("Input folder '{}' does not exist or is not a directory".format(i))
        sys.exit()
    
    if not(os.path.exists(m)) or not(os.path.isfile(m)):
        print("Model file '{}' does not exist or is not a file".format(m))
        sys.exit()

    if not(os.path.exists(o)):
        print("Output folder '{}' does not exist. Creating it...".format(o))
        os.makedirs(o)
        
    # Overall params
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
                "optimizer"        : "Adam",
                "lr"               : 1e-4,
                "weight_decay"     : 1e-5,
                "lr_scheduling"    : "step",
                "lr_gamma"         : 0.1,
                "step"             : 15, 
                "plateau_mode"     : "min", 
                "plateau_patience" : 10,
                "var_beta"         : 1,
                "loss_reduction"   : "mean",
                "metadata" : False,
                "K" : 1}  

    
    aug = augmentation(augmentation_params) # Augmentation class
    preproc = preprocessing(preprocessing_params) # Preprocessing class
    
    # General arguments for data loading
    args = [os.path.join(i,"overview.csv"),
            os.path.join(i,"dicom_dir"),
            preproc,
            True,
            aug]
    
    trainValData, testData = Images(*args).splitTrainTest(0.85)

    # Get architecture
    net = VAE(model_params,1,1,True)
    
    # Load trained checkpoint
    optimizer,scheduler = train.optimizerExtractor(net,model_params)
    net,optimizer,best_ssim = utilities.model_loading(net,optimizer,m)
    
    # Define loss function
    loss_fun = utilities.vae_loss(model_params)
        
    test_loader = getDataLoader(testData,model_params)
    
    # Set device
    device = torch.device("cuda:0" if model_params["use_gpu"] and torch.cuda.is_available() else "cpu")
    
    net = net.to(device)
    
    # Define output files
    out_info_file = os.path.join(o,"out_metrics_latentDims{}_lr{}_epochs{}.txt".format(model_params["latent_dims"],
                                                                                       model_params["lr"],
                                                                                       model_params["epochs"]))
    out_latent_file = os.path.join(o,"out_latent_latentDims{}_lr{}_epochs{}.txt".format(model_params["latent_dims"],
                                                                                       model_params["lr"],
                                                                                       model_params["epochs"]))
    
    plot_folder = os.path.join(o,"Test_plots")
    if not(os.path.exists(plot_folder)):
        os.makedirs(plot_folder)
    
    latent_spaces = [] # Save latent space vectors of test set in this list
    
    maes = [] # Store MAEs for all test samples
    ssims = [] # Store SSIMs for all test samples
    psnrs = [] # Store PSNRs for all test samples
    losses = [] # Store losses for all test samples
    ages = [] # Store read ages
    csts = [] # Store read contrasts
    tags = [] # Store read tags
    
    cont_sample = 0
    
    # Get information from VAE with all test samples
    
    with torch.no_grad():
        net.eval() 
        
        for enum_loader,batch in enumerate(test_loader):
            
            print("Inference, sample {}\n".format(cont_sample))
            
            # Feed batch to model
            recon_batch,mu_batch,logvar_batch,latent_var = net(batch,device)
            
            latent_spaces.append(torch.squeeze(latent_var).cpu().numpy())
            
            ages.append(int(batch["AGE"]))
            csts.append(float(batch["CST"]))
            tags.append(int(batch["TAG"]))
            
            batch = batch["DCM"].to(device)
            
            # Loss
            loss = loss_fun(recon_batch, batch, mu_batch, logvar_batch)
            losses.append(loss.item())
            
            # SSIM and PSNR metrics
            mae,ssim,psnr = utilities.metrics(recon_batch,batch)
            maes.append(mae)
            ssims.append(ssim)
            psnrs.append(psnr)
            
            # Plotting
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(torch.squeeze(batch).cpu(),cmap="gray")
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(torch.squeeze(recon_batch).cpu(),cmap="gray")
            plt.colorbar()
            fig.savefig(os.path.join(plot_folder,"test_plot_sample{}_age{}_cst{}_tag{}.png".format(cont_sample,ages[-1],int(csts[-1]),tags[-1])))
            
            cont_sample += 1
    
    # Write output information about VAE training
    # Information about metrics in the test set
    with open(out_info_file,"w") as f:
        f.write("Test sample,Loss,MAE,SSIM,PSNR,age,cst,tag\n")
        cont_sample = 0
        for l,mae,ssim,psnr,age,cst,tag in zip(losses,maes,ssims,psnrs,ages,csts,tags):
            f.write("{},{},{},{},{},{},{},{}\n".format(cont_sample,l,np.array(mae).mean(),np.array(ssim).mean(),np.array(psnr).mean(),age,cst,tag))
            cont_sample += 1
        f.close()
    
    # Save produced latent spaces from test samples    
    latent_spaces = np.array(latent_spaces).astype(float)
    np.savetxt(out_latent_file,latent_spaces,fmt="%f")
    
    print("Inference completed! Time ellapsed: {}sec ({}sec/sample)".format(round(time.time()-init,2),round((time.time()-init)/cont_sample,2)))



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
                        shuffle=False,
                        pin_memory=True)
    
    return loader

if __name__ == "__main__":
    main(sys.argv[1:])  