#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:38:43 2019

@author: luxemk
"""

import os
import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import torch.utils.data as Data
from torch.autograd import Variable

from VAME.util.auxiliary import read_config
from VAME.model.spatial.VAE import VAE
from VAME.model.spatial.VAE import resnet
from VAME.model.temporal.LSTM_VAE import RNN_VAE
#from VAME.model.temporal.RNN_VAE_auxiliary import RNN_VAE
#from VAME.model.temporal.RNN_Ablation import RNN_VAE
from VAME.model.dataloader.spatial_dataloader import VIDEO_DATASET
from VAME.model.dataloader.temporal_dataloader import SEQUENCE_DATASET 
from VAME.model.dataloader.spatial_features_dataloader import SPATIAL_FEATURES
    

def generate_reconstructions(model, test_loader, PROJECT, PROJECT_PATH):
    model.eval()
    x = test_loader.__iter__().next()
    x = x[:24].type('torch.FloatTensor').cuda()
    x_tilde, _, = model(x)

    x_cat = torch.cat([x, x_tilde], 0)
    images = (x_cat.cpu().data + 1) / 2
    
    fig, ax = plt.subplots()
    ax.imshow(x[0,0,...].cpu().detach().numpy(),cmap='gist_gray')
    ax.grid(False)
    fig, ax = plt.subplots()
    ax.imshow(x_tilde[0,0,...].cpu().detach().numpy(),cmap='gist_gray')
    ax.grid(False)
    
    save_image(
        images,
        PROJECT_PATH+'/model/spatial_model/evaluate/'+'vae_reconstructions_{}.eps'.format(PROJECT),
        nrow=10
    )
    
    
def plot_loss(model_name, train_loss, test_loss, kl_losses, weight_values, filepath, log_scale=False):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Train Loss vs. Test Loss and KL-Loss vs. KL-Weight')
    ax1.set(xlabel='Epochs', ylabel='loss [%]')
    ax1.plot(train_loss, 'b', label='train_loss')
    if log_scale:
        ax1.set_yscale("log")
    ax1.plot(test_loss, 'r', label='test_loss')
    ax2.plot(kl_losses, 'b', label='kl_loss')
    if log_scale:
        ax2.set_yscale("log")
    ax2.set(xlabel='Epochs', ylabel='KL-Loss')
    ax1.legend()
    ax2.legend()
    fig.savefig(filepath+'evaluate/'+'MSE-and-KL-Loss'+model_name+'.png')
    
    
def eval_temporal(cfg, use_gpu, model_name):
    
    SEED = 19
    ZDIMS = cfg['ZDIMS_temporal']
    FUTURE_DECODER = cfg['future_decoder']
    TEMPORAL_WINDOW = cfg['temporal_window']
    FUTURE_STEPS = TEMPORAL_WINDOW - cfg['future_steps']
    NUM_FEATURES = cfg['num_features']
    TEST_BATCH_SIZE = 64
    PROJECT_PATH = cfg['project_path']
    FUTURE_DECODER = cfg['future_decoder']
    
    filepath = PROJECT_PATH+'model/temporal_model/'

    seq_len_half = int(TEMPORAL_WINDOW/2)
    if use_gpu:
        torch.cuda.manual_seed(SEED)
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS).cuda()
    else:
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS)
        
    model.load_state_dict(torch.load(cfg['project_path']+'/'+'model/temporal_model/best_model/snapshots/'+model_name+'_'+cfg['Project']+'_epoch_50.pkl'))
    model.eval() #toggle evaluation mode
    
    testset = SEQUENCE_DATASET(cfg['project_path']+'data/train/temporal/', data='sequence_test_full.npy', train=False, temporal_window=TEMPORAL_WINDOW)
    test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)
    
    train_loss = np.load(cfg['project_path']+'/'+'model/temporal_model/losses'+'/train_losses'+model_name+'.npy')
    test_loss = np.load(cfg['project_path']+'/'+'model/temporal_model/losses'+'/test_losses'+model_name+'.npy')
#    mse_loss = np.load(cfg['project_path']+'/'+'model/temporal_model/losses'+'/kmeans_losses'+model_name+'.npy')
    kl_loss = np.load(cfg['project_path']+'/'+'model/temporal_model/losses'+'/kl_losses'+model_name+'.npy')
    weight_values = np.load(cfg['project_path']+'/'+'model/temporal_model/losses'+'/weight_values'+model_name+'.npy')
    
    train_loss = train_loss / np.max(train_loss)
    test_loss = test_loss / np.max(test_loss)
    kl_losses = kl_loss #/ np.max(kl_loss)
    weight_values = weight_values / np.max(weight_values)
    
#    mse_loss = mse_loss / train_loss
    kl_loss = kl_losses / train_loss
    
    x = test_loader.__iter__().next()
    x = x.permute(0,2,1)
    data = x[:,:seq_len_half,:].type('torch.FloatTensor').cuda()
    data_fut = x[:,seq_len_half:45,:].type('torch.FloatTensor').cuda()
    if FUTURE_DECODER:
        x_tilde, future, latent, mu, logvar = model(data)
        
        fut_orig = data_fut.cpu()
        fut_orig = fut_orig.data.numpy()
        fut = future.cpu()
        fut = fut.detach().numpy()
    
    else:
        x_tilde, latent, mu, logvar = model(data)

    data_orig = data.cpu()
    data_orig = data_orig.data.numpy()
    data_tilde = x_tilde.cpu()
    data_tilde = data_tilde.detach().numpy()
    
    if FUTURE_DECODER:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Reconstruction and future prediction of input sequence')
        ax1.plot(data_orig[1,...], color='k', label='Sequence Data')
        ax1.plot(data_tilde[1,...], color='r', linestyle='dashed', label='Sequence Reconstruction') 
        ax2.plot(fut_orig[1,...], color='k')
        ax2.plot(fut[1,...], color='r', linestyle='dashed')
        fig.savefig(filepath+'evaluate/'+'Future_Reconstruction.png') 
    
    else:
        fig, ax1 = plt.subplots(1, 1)
        fig.suptitle('Reconstruction of input sequence')
        ax1.plot(data_orig[1,...], color='k', label='Sequence Data')
        ax1.plot(data_tilde[1,...], color='r', linestyle='dashed', label='Sequence Reconstruction') 

        fig.savefig(filepath+'evaluate/'+'Reconstruction_'+model_name+'.png') 
        
    plot_loss(model_name, train_loss, test_loss, kl_losses, weight_values, filepath)
    
    return 
    
    
def evaluate_model(config, model_name):
    """
        Evaluation of testset
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    if not os.path.exists(cfg['project_path']+'model/temporal_model/evaluate/'):
        os.mkdir(cfg['project_path']+'/'+'model/temporal_model/evaluate/')
            
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
        print('GPU active:',torch.cuda.is_available())
        print('GPU used:',torch.cuda.get_device_name(0)) 
    else:
        print("CUDA is not working!")
      
    print("\n\nEvaluation of %s model. \n" %model_name)   
    eval_temporal(cfg, use_gpu, model_name)

    print("You can find the results of the evaluation in the '/model/temporal_model/evaluate/' folder \n"
          "OPTIONS:\n" 
          "- vame.cluster_behavior() to identify behavioral motifs.\n"
          "- re-run the model for further fine tuning. Check again with vame.evaluate_model()")
    
    
    
    
    