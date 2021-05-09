Code for Automatic and Unsupervised Extraction of Features from lung CT scans with a convolutional Variational AutoEncoder (VAE), with the encoder pretrained on VGG-11 network on ImageNet

Dataset used: https://www.kaggle.com/kmader/siim-medical-images

Code description:

- "cross_validation.py" and "andres_training.ipynb": code for k-fold cross-validation, includes parameters to be changed while training, and main folders where to access data, 
and save intermediate results. The .py file contains the pure Python code that can be run in a terminal, while the .ipynb is a notebook that was used for training in Google Colab.
- "dataLoader.py": code for preprocessing, augmenting, and loading the images from the Lung CT dataset from Kaggle. 
- "model_vgg11_shortSkipPretrained.py": .py file with a four-layer VAE whose encoder had been pretrained on ImageNet (VGG-11 network)
- "train.py": code for training, either with k-fold cross-validation procedure, or with an only train dataset. Models are saved always that the validation SSIM is improved during training.
- "evaluate.py": code for evaluating a validation set of lung CT images during k-fold cross-validation in terms of smooth L1-L2 loss +Kullback Leibler loss, SSIM, PSNR, 
   and Mean Squared Error.
- "utilities.py": code with loss function definition (smooth L1-L2 loss + Kullback-Leibler), metric computation (MSE,SSIM,PSNR), and model saving and loading functions
- "inference.py": code for making inference with a test dataset, and a saved model checkpoint during training. Can be run from the terminal as:
	python inference.py -i <data_folder> -m <model_checkpoint_file> -o <output_results_folder>
	The results outputted are the metrics for each test sample in one TXT file, the comparison of the test sample with its reconstruction, and the latent space for each sample in 
	another TXT file. All results are outputted to a specified output folder.




Libraries required: torch, Albumentations, Pydicom, elasticdeform, numpy, os, sys, matplotlib, itertools, scipy, scikit-learn