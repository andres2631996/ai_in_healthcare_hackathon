from torch import nn
import torch
from torchvision import models
import numpy as np

###
# Original implementation courtesy of:
# Dag Lindgren
# Andreas Wallin
# Lowe Lundin

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

def conv1x1(in_, out):
    return nn.Conv2d(in_, out, 1, padding=0)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, batch_norm):
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels,middle_channels,batch_norm),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ConvRelu(nn.Module):
    def __init__(self, in_, out, batch_norm):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)
        self.batch_norm = batch_norm

    def forward(self, x):
        x = self.conv(x)
        
        if self.batch_norm:
            x = self.bn(x)
        
        x = self.activation(x)
        return x

class VAE(nn.Module):
    def __init__(self, params, channel_count=3, class_count=1, pre_trained=True):

        """
        Params:
            - filters : int
                Number of initial filters for feature extraction
            - latent_dims : int
                Number of latent space dimensions
            - output_dim : int
                Output dimensions
            - train : bool
                Whether the model is being trained or not
            - channel_count : int
                Input number of channels (default: 3)
            - class_count : int
                Output number of classes (default: 1)
            - pre_trained : bool
                Flag for using or not transfer-learning , with VGG-11 from ImageNet
            - train : bool
                Flag for stating whether the model is trained or not

        """
        
        super(VAE, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.convertconv = conv1x1(in_=channel_count, out=3)
        self.channels = channel_count
        self.params = params

        # Pretrained encoder layers
        encoder = models.vgg11(pretrained=pre_trained).features

        self.relu = encoder[1]
        self.conv1 = encoder[0]
        self.conv2 = encoder[3]
        self.conv3s = encoder[6]
        self.conv3 = encoder[8]
        self.conv4s = encoder[11]
        self.conv4 = encoder[13]
        self.conv5s = encoder[16]
        self.conv5 = encoder[18]
        
        self.downsample2 = nn.Conv2d(in_channels = self.params["filters"] * 2, 
                                     out_channels = self.params["filters"] * 2 * 2, 
                                     kernel_size =1)
        self.downsample3 = nn.Conv2d(in_channels = self.params["filters"] * 2 * 2, 
                                     out_channels = self.params["filters"] * 2 * 4, 
                                     kernel_size = 1)
        self.downsample4 = nn.Conv2d(in_channels = self.params["filters"] * 2 * 4, 
                                     out_channels = self.params["filters"] * 2 * 8, 
                                     kernel_size = 1)
        self.downsample5 = nn.Conv2d(in_channels = self.params["filters"] * 2 * 8, 
                                     out_channels = self.params["filters"] * 2 * 8, 
                                     kernel_size =1)

        self.bn1 = nn.BatchNorm2d(self.params["filters"]*2)
        self.bn2 = nn.BatchNorm2d(self.params["filters"]*4)
        self.bn3 = nn.BatchNorm2d(self.params["filters"]*8)
        self.bn4 = nn.BatchNorm2d(self.params["filters"]*16)
        self.bn5 = nn.BatchNorm2d(self.params["filters"]*32)
        

        # Decoder layers
        self.center = DecoderBlock(self.params["filters"] * 8 * 2, self.params["filters"] * 8 * 2, self.params["filters"] * 8, self.params["batch_norm"])
        self.dec5 = DecoderBlock(self.params["filters"] * 4 * 2, self.params["filters"] * 4 * 2, self.params["filters"] * 4, self.params["batch_norm"])
        self.dec4 = DecoderBlock(self.params["filters"] * 2 * 2, self.params["filters"] * 2 * 2, self.params["filters"] * 2, self.params["batch_norm"])
        self.dec3 = DecoderBlock(self.params["filters"] * 2, self.params["filters"] * 2, self.params["filters"], self.params["batch_norm"])
        self.dec2 = DecoderBlock(self.params["filters"], self.params["filters"], self.params["filters"], self.params["batch_norm"])

        # Final layer
        self.final = nn.Conv2d(in_channels=self.params["filters"], out_channels=class_count, kernel_size=1)

        # Latent space layers
        self.fc_mu = nn.Linear(int((2**(self.params["layers"]-1))*self.params["filters"]*(self.params["output_dim"]/(2**self.params["layers"]))**2),
                               int(self.params["latent_dims"]))
        self.fc_logvar = nn.Linear(int((2**(self.params["layers"]-1))*self.params["filters"]*(self.params["output_dim"]/(2**self.params["layers"]))**2),
                                   int(self.params["latent_dims"]))
        
        if self.params["metadata"]:
            self.fc = nn.Linear(int(self.params["latent_dims"])+3,
                            int((2**(self.params["layers"]-1))*self.params["filters"]*(self.params["output_dim"]/(2**self.params["layers"]))**2))
        else:
            self.fc = nn.Linear(int(self.params["latent_dims"]),
                                int((2**(self.params["layers"]-1))*self.params["filters"]*(self.params["output_dim"]/(2**self.params["layers"]))**2))
       
        
    def latent_sample(self, mu, logvar):
        """
        Sample latent space
        
        Params:
            - mu : PyTorch tensor
                Vector of means
            - logvar : Pytorch tensor
                Vector of logged variances
        
        """
        
        if self.training:
            # the reparametrization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


    def forward(self, x, device):
        """
        Input sample in network
        
        Params:
            - x : PyTorch tensor
                Input CT sample
            - device : str
                Device where to train model: GPU if available, else CPU
                  
        Returns:
            - out : PyTorch tensor
                Reconstructed CT sample
            - sample_mu : PyTorch tensor
                Sampled mean from latent space
            - sample_logvar : PyTorch tensor
                Sampled variance from latent space

        """
        
        
        # Process input according to number of input channels 
        
        if self.params["metadata"]:
            meta = torch.tensor([np.array(x["AGE"]),
                                 np.array(x["CST"]).astype(float),
                                 np.array(x["TAG"])]).to(device).float()
            meta = meta.view(-1,meta.shape[0])
        
        x = x["DCM"].to(device).float()
        
        if self.channels == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.channels == 2:
            x = torch.cat((x, x[:, :1, :, :]), 1)
        if self.channels > 3: 
            x = self.convertconv(x)

        # Encoder layers 
        if self.params["batch_norm"]:
            conv1 = self.relu(self.bn1(self.conv1(x)))
            conv1p = self.pool(conv1)
    
            conv2 = self.relu(self.bn2(self.conv2(conv1p) + self.downsample2(conv1p)))
            conv2p = self.pool(conv2)
    
            conv3s = self.relu(self.bn3(self.conv3s(conv2p)))
            conv3 = self.relu(self.bn3(self.conv3(conv3s) + self.downsample3(conv2p)))
            conv3p = self.pool(conv3)
    
            conv4s = self.relu(self.bn4(self.conv4s(conv3p)))
            conv4 = self.relu(self.bn4(self.conv4(conv4s) + self.downsample4(conv3p)))
            conv4p = self.pool(conv4)
    
            conv5s = self.relu(self.bn4(self.conv5s(conv4p)))
            conv5 = self.relu(self.bn4(self.conv5(conv5s) + self.downsample5(conv4p)))
            conv5p = self.pool(conv5)
  
        else:
            conv1 = self.relu(self.conv1(x))
            conv1p = self.pool(conv1)
    
            conv2 = self.relu(self.conv2(conv1p) + self.downsample2(conv1p))
            conv2p = self.pool(conv2)
    
            conv3s = self.relu(self.conv3s(conv2p))
            conv3 = self.relu(self.conv3(conv3s) + self.downsample3(conv2p))
            conv3p = self.pool(conv3)
    
            conv4s = self.relu(self.conv4s(conv3p))
            conv4 = self.relu(self.conv4(conv4s) + self.downsample4(conv3p))
            conv4p = self.pool(conv4)
    
            conv5s = self.relu(self.conv5s(conv4p))
            conv5 = self.relu(self.conv5(conv5s) + self.downsample5(conv4p))
            conv5p = self.pool(conv5)
        
        # Latent space layers
        sample = conv5p.view(conv5p.size(0), -1)
        
        sample_mu = self.fc_mu(sample)
        sample_logvar = self.fc_logvar(sample)
        
        latent_var = self.latent_sample(sample_mu,sample_logvar)
        if self.params["metadata"]:
            latent_var = torch.cat((latent_var,meta),1)
        latent_var_decoder = self.fc(latent_var)
        latent_var_decoder = latent_var_decoder.view(latent_var_decoder.size(0), 
                                     int(self.params["filters"]*(2**(self.params["layers"]-1))), 
                                     int(x.shape[-2]/(2**self.params["layers"])), 
                                     int(x.shape[-1]/(2**self.params["layers"])))

        # Decoder layers
        center = self.center(latent_var_decoder)

        dec5 = self.dec5(center)
        dec4 = self.dec4(dec5)
        dec3 = self.dec3(dec4)
        dec2 = self.dec2(dec3)
#        dec1 = self.dec1(dec2)
        return self.final(dec2),sample_mu,sample_logvar,latent_var