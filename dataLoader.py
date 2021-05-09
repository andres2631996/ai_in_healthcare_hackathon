import cv2
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import albumentations as A
import os
import elasticdeform
import random
import numpy as np
from numpy.random import randint
import pickle
import pydicom as dcm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


__all__ = ["preprocessing", "augmentation", "Images"]

class preprocessing:
    """
    Preprocess individual 2D CT slices (resampling (optional) + histogram equalization (optional) 
    + normalization (optional))
    
    Params:
        - img : np.ndarray
            Slice to preprocess
        - res : float or int
            Corresponding image resolution
        - params : dictionary
            Set of preprocessing parameters

    Output:
        - out : np.ndarray
            Preprocessed image

    """   

    def __init__(self,params):
        self.params = params
        
    def imageResampling(self):
        """
        Resample image to specified dimensions in General parameter file
    
        Outputs
            - final_img : np,ndarray
                Output resampled slice

        """

        if self.params["resample"]:
            # Get resampled image
            if len(self.img.shape) == 3:
                img_r = zoom(self.img,(1,self.res/self.params["resolution"],
                                       self.res/["params.resolution"]),
                             order=self.params["resample_order"],mode="constant",
                             cval=self.img.flatten().min())
                
            elif len(self.img.shape) == 2:
                img_r = zoom(self.img,self.res/self.params["resolution"],order=self.params["resample_order"],
                             mode="constant",cval=self.img.flatten().min())        
        else:            
            img_r = self.img.copy()
        
        # Crop or pad image to fit desired dimensions
        if img_r.shape[1] > self.params["output_dim"]: # Cropping
            
            # Crop image around the center
            if len(self.img.shape) == 3:
                final_img = img_r[:,(img_r.shape[1]//2-self.params["output_dim"]//2):(img_r.shape[1]//2+self.params["output_dim"]//2),
                                        (img_r.shape[2]//2-self.params["output_dim"]//2):(img_r.shape[2]//2+self.params["output_dim"]//2)]
            elif len(self.img.shape) == 2:
                final_img = img_r[((img_r.shape[0]//2)-(self.params["output_dim"]//2)):((img_r.shape[0]//2)+(self.params["output_dim"]//2)),
                                        ((img_r.shape[1]//2)-(self.params["output_dim"]//2)):((img_r.shape[1]//2)+(self.params["output_dim"]//2))]
            
        elif img_r.shape[1] < self.params["output_dim"]: # Padding

            if len(self.img.shape) == 3:
                diff = (self.params["output_dim"]-img_r.shape[1],self.params["output_dim"]-img_r.shape[2]) 
                if img_r.shape[1]%2 == 0:
                    final_img = np.pad(img_r,((0,0),(diff[0]//2,diff[0]//2),(diff[1]//2,diff[1]//2)),
                                       mode="minimum")
                else:
                    final_img = np.pad(img_r,((0,0),(diff[0]//2+1,diff[0]//2),(diff[1]//2+1,diff[1]//2)),
                                       mode="minimum")
            elif len(self.img.shape) == 2:
                diff = (self.params["output_dim"]-img_r.shape[0],self.params["output_dim"]-img_r.shape[1]) 
                if img_r.shape[0]%2 == 0:
                    final_img = np.pad(img_r,((diff[0]//2,diff[0]//2),(diff[1]//2,diff[1]//2)),
                                       mode="minimum")
                else:
                    final_img = np.pad(img_r,((diff[0]//2+1,diff[0]//2),(diff[1]//2+1,diff[1]//2)),
                                       mode="minimum")
            
        else:
            final_img = img_r.copy()

        return final_img
    
    
    def image_histogram_equalization(self, image):
        """
        Perform CLAHE equalization
        
        Params:
            - image : np.ndarray
                Image to be histogram-equalized
                
        Outputs:
            - image_equalized : np.ndarray
                Result of histogram equalization
         
        """
        
        winSize = (2 ** self.params["clahe_win"], 2 ** self.params["clahe_win"])
        #clip_size = int((4 ** self.params["clahe_win"]) / 3)
        clip_size = self.params["clahe_clip_size"]
        clahe = cv2.createCLAHE(clipLimit = clip_size, tileGridSize = winSize)
        
        min_img = np.amin(image)
        max_img = np.amax(image)
        image_norm = ((image - min_img)*255/(max_img - min_img)).astype(np.uint8) # Normalization from 0 to 1 for OpenCV
        
        if len(image.shape) == 2: # 2D image        
            image_equalized = clahe.apply(image_norm)
        elif len(image.shape) == 3: # 2.5D image            
            image_equalized = np.zeros(image.shape)

            for z in range(image.shape[0]):
                image_equalized[z] = clahe.apply(image_norm[z])
        
        image_equalized = image_equalized*(max_img - min_img)/255 + min_img # Return image to original range
        return image_equalized    

     
    def __call__(self, img, res):
        self.img = img
        self.res = res
        
        # Resampling        
        img_r = self.imageResampling()
        max_img = np.percentile(img_r.flatten(),99)
        min_img = np.percentile(img_r.flatten(),1)
        
        # Outlier clipping
        img_clip = np.clip(img_r,min_img,max_img)
        
        # Histogram equalization: apply it only in the lungs
        if self.params["equalize"]:
            img_e = self.image_histogram_equalization(img_clip)

#            img_norm = (img_clip - min_img)/(max_img - min_img)
#            img_e = equalize_adapthist(img_norm)
#            img_e = min_img + (max_img - min_img)*img_norm
            
        else:
            img_e = img_clip.copy()
        
        # Normalization
        if self.params["normalize"]:
            out = (img_e - img_e.flatten().mean())/(img_e.flatten().std() + np.finfo(float).eps)
        else:
            out = img_e.copy()
        
        return out


class augmentation:
    """
    Perform augmentation with Albumentations library, together with parameters given in 
    main parameter file
    
    Potential transformations: flip + rotation + scaling + elastic deformation + contrast + brightness

    Params:
        - img : np.ndarray
            Image to be augmented
    
    Outputs:
        - out : np.ndarray
            Augmented image

    """        
    
    def __init__(self, params):
        self.params = params
        
        
    def __call__(self, img):        
        self.img = img
        
        min_val = torch.min(torch.flatten(self.img))

        if self.img.shape[0] == 1: # 2D images
            self.img = torch.squeeze(self.img)
            pipeline = A.Compose([A.Flip(p = 0.5),
                                  A.Rotate(self.params["rot"],
                                        interpolation = 1,
                                        border_mode = 0,
                                        value = min_val.item(),
                                        p = 0.5),
                                  A.RandomBrightnessContrast(brightness_limit = self.params["bright"],
                                                             contrast_limit = self.params["contrast"],
                                                             p = 0.5)],    
                                 p = 0.5)
            
            data = {"image": np.float32(self.img)}
            out = pipeline(**data)
            out = np.expand_dims(out["image"],0)

                         
        elif self.img.shape[0] > 1: # 2.5D images
            pipeline = A.Compose(
                        [A.Flip(p=0.5),
                         A.RandomBrightnessContrast(brightness_limit=self.params["bright"],
                                                    contrast_limit=self.params["contrast"],
                                                    p=0.5)],    
                        p = 0.75)
                     
            data = {"image": np.float32(self.img)}
            out = pipeline(**data)
            out = out["image"]
        
        # Elastic deformations with library elasticdeform
        if random.uniform(0,1) < 0.5:
            elasticdeform.deform_random_grid(out, self.params["sigma"], 
                                             self.params["points"], 
                                             order=self.params["resample_order"])
         
        return torch.from_numpy(out).float()
    
    

#####################################################################################################
#####################################################################################################
#####################################################################################################
        
class Images (Dataset, object):
    """
        To load the data. It links the records from the overview.csv file with the 
        dicom images. 
    
        Params:
            - csv_path  : str (default None)
                          Path of the overview.csv file

            - dcm_path  : str (default None)
                          Path of the folder containing all the dicom images

            - preproc   : callable (default None)
                          Object of type "preprocessing". If given, it will apply the
                          pre-processing on the data during the loading. If None, no 
                          pre-processing methods will be applied

            - use_HU    : bool (default True)
                          To enable (if True) or disable (if False) the conversion
                          of the image to Hounsfield Unit. The conversion is an 
                          affine function defined as "RescaleSlope * image + RescaleIntercept",
                          where RescaleSlope and RescaleIntercept are two attributes
                          of DICOM header

            - transform : callable (default None)
                          Object of type "augmentation". If given, the transforms
                          contained in @transform are applied when returning an
                          element of @Images. If None, no data augmentation will
                          be applied

    """

    def __init__(self, csv_path = None, dcm_path = None, preproc = None, use_HU = True, transform = None):
        super().__init__()
        
        self.__dcm = []
        self.__res = []
        self.__age = []
        self.__contrast = []
        self.__contrastTag = []
        
        if (csv_path and dcm_path):																			# loads the data if paths are given
            with open(csv_path, "r") as f:																	# opens the file "overview.csv"
                skip = True																					# flag for Header
                
                for line in f:
                    if (skip):																				# skips the header
                        skip = False
                        
                    else:
                        tmp = line.split(",")																# splits the table record using the comma
                        
                        self.__age.append(int(tmp[1]))														# appends the age
                        self.__contrast.append(bool(tmp[2]))												# appends the contrast
                        self.__contrastTag.append(tmp[3])													# appends the contrast tag
                        
                        dcm_im = dcm.dcmread(os.path.join(dcm_path,tmp[-1].replace("\n", "")))							# gets the DICOM image
                        
                        # hu_converter : lambda function that converts the image into Hounsfield
                        # Unit if @use_HU is equal to True, otherwise it returns the image itself
                        # with no conversion
                        hu_converter = \
                            lambda x : float(dcm_im.RescaleSlope) * x + float(dcm_im.RescaleIntercept) \
                            if use_HU \
                            else x
                            
                        if (preproc):																		
                            dcm_im = preproc(hu_converter(dcm_im.pixel_array), dcm_im.PixelSpacing[0])		# applies the pre-processing if @preproc is not None
                        else:
                            dcm_im = hu_converter(dcm_im.pixel_array)										# skips the pre-processing if @preproc is None
                            
                        self.__dcm = np.expand_dims(dcm_im, 0) if len(self.__dcm) == 0 \
                            else np.append(self.__dcm, np.expand_dims(dcm_im, 0), 0)				        # appends the dicom image
                            
            self.__dcm = torch.from_numpy(np.expand_dims(self.__dcm, 1))									# final shape of DICOM: (100,1,512,512)
            self.__age = torch.from_numpy(np.expand_dims(np.array(self.__age), 0))							# final shape of AGE: (1,100)
            self.__contrast = torch.from_numpy(np.expand_dims(np.array(self.__contrast), 0))				# final shape of CONTRAST: (1,100)
            
            tmp = sorted(set(self.__contrastTag))															# set of unique sorted contrast tags
            self.__tag2num = { tag : i for i, tag in enumerate(tmp)}										# "tag to number" dictionary, to convert the tag into number
            self.__num2tag = { i : tag for i, tag in enumerate(tmp)}										# "number to tag" dictionary, to convert the number into tag
            
            self.__contrastTag = np.array([self.__tag2num[i] for i in self.__contrastTag])
            self.__contrastTag = torch.from_numpy(np.expand_dims(self.__contrastTag, 0))					# final shape of CONTRAST TAG: (1,100)
            
        self.__tran = transform																				# transform to perform data augmentation
        

    @property
    def dcms(self):
        return self.__dcm
        

    @property
    def ages(self):
        return self.__age
        

    @property
    def contrasts(self):
        return self.__contrast
        

    @property
    def tags(self):
        return self.__contrastTag
        

    @property
    def tag2num(self):
        return self.__tag2num
        

    @property
    def num2tag(self):
        return self.__num2tag
    
    @property
    def trans(self):
        return self.__tran
        

    def __len__(self):
        return self.__dcm.shape[0]
        

    def __getitem__(self, idx):
        dcm = self.__tran(self.__dcm[idx,:,:,:]) if self.__tran else self.__dcm[idx,:,:,:]                  # applies the transform if any
        age = self.__age[:,idx]
        cst = self.__contrast[:,idx]
        tag = self.__contrastTag[:,idx]
        
        return { "DCM" : dcm, "AGE" : age, "CST" : cst, "TAG" : tag }
                 

    def save(self, path):
        with open(path, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            
            return True
            

    def splitTrainTest(self, trainSize): 
        """
            Splits the dataset into training set and testing set, according to the
            @trainSize.
            
            Params:
                - trainSize : float or tuple
                              Size of the training set. Available values are (0,1)
                              with ends exclueded.
                              If tuple : tuple of indices for train and validation : (train_idx,val_idx)
                              
            Output:
                - trainData : Images
                              Object of type "Images" containing the training data.
                              Transforms are copied if present

                - testData  : Images
                              Object of type "Images" containing the training data.
                              Transforms are NOT copied if any case
        """
        
        if isinstance(trainSize,float): # Sample a random fraction of the dataset
            
            if (trainSize <= 0 or trainSize >= 1):
                print("Availables value for @trainSize parameter are from 0 to 1, both excluded.")
                return

            n = list(range(self.__len__()))
            
            trainIndex = np.random.choice(n, int(self.__len__()*trainSize), replace = False)	
            testIndex = list(set(n) - set(trainIndex))
            
        elif isinstance(trainSize,tuple): # Sample training and validation indices from dataset
            
            trainIndex = trainSize[0]
            testIndex = trainSize[1]
            
        trainData = Images()
        trainData.__dcm = self.__dcm[trainIndex,:,:]
        trainData.__age = self.__age[:,trainIndex]
        trainData.__contrast = self.__contrast[:,trainIndex]
        trainData.__contrastTag = self.__contrastTag[:,trainIndex]
        trainData.__tran = self.__tran
        
        testData = Images()
        testData.__dcm = self.__dcm[testIndex,:,:]
        testData.__age = self.__age[:,testIndex]
        testData.__contrast = self.__contrast[:,testIndex]
        testData.__contrastTag = self.__contrastTag[:,testIndex]
        testData.__tran = None
        
        return trainData, testData
        
        
    @staticmethod
    def load(path):
        dataset = None
        with open(path, "rb") as input:
            dataset = pickle.load( input )
            
        return dataset

#####################################################################################################
#####################################################################################################
#####################################################################################################

def plotItem(item):
    plt.figure()

    plt.imshow(np.squeeze(item["DCM"]), cmap = "gray")
    #plt.title("Dicom image")

    plt.suptitle(f"Age: {item['AGE']}\nContrast: {'APPLIED' if item['CST'] == 0 else 'NOT APPLIED'} - Contrast Tag: {num2tag[item['TAG'].item()]}")
    plt.colorbar()
    plt.show()


# Global variables

#path = r"C:\Users\andre\Downloads\kaggle_data"              # TO ACCESS THIS FOLDER: 
#                                                                                # RIGHT CLICK IN AI_in_Healthcare_Hackathon --> ADD SHORTCUT TO DRIVE FOLDER 
#                                                                                # (OR STH LIKE THAT IN ITALIAN) --> SELECT MY DRIVE
##path_dicom = path + "images/"
##pat_results = path + "results/"
#
#if __name__ == "__main__":
#    preprocessing_params = { "resample"       : True,
#                             "resolution"     : 1.2,
#                             "output_dim"     : 256,
#                             "resample_order" : 2,
#                             "equalize"       : True,
#                             "clahe_win"      : 2,
#                             "clahe_clip_size": 5,  
#                             "normalize"      : True}
#
#    augmentation_params = { "rot"             : 60,
#                            "contrast"        : 0.02, 
#                            "bright"          : 0.02,
#                            "sigma"           : 2,
#                            "points"          : 8,
#                            "resample_order"  : 2}
#
#    aug = augmentation(augmentation_params)
#    preproc = preprocessing(preprocessing_params)
#
#    args = [os.path.join(path,"overview.csv"),
#            os.path.join(path,"dicom_dir"),
#            preproc,
#            True,
#            aug]
#
#    dataset = Images(*args)
#    print(dataset[25]["DCM"].shape)
#    plotItem(dataset[25])
    