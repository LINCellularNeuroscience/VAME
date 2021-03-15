#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch.utils.data as Data

from vame.util.auxiliary import read_config
from vame.model.rnn_vae import RNN_VAE
from vame.model.dataloader import SEQUENCE_DATASET

use_gpu = torch.cuda.is_available()
if use_gpu:
    pass
else:
    torch.device("cpu")


def plot_reconstruction(filepath, test_loader, seq_len_half, model, model_name,
                        FUTURE_DECODER, FUTURE_STEPS):
    x = test_loader.__iter__().next()
    x = x.permute(0,2,1)
    if use_gpu:
        data = x[:,:seq_len_half,:].type('torch.FloatTensor').cuda()
        data_fut = x[:,seq_len_half:seq_len_half+FUTURE_STEPS,:].type('torch.FloatTensor').cuda()
    else:
        data = x[:,:seq_len_half,:].type('torch.FloatTensor').to()
        data_fut = x[:,seq_len_half:seq_len_half+FUTURE_STEPS,:].type('torch.FloatTensor').to()
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
        fig.savefig(os.path.join(filepath,"evaluate",'Future_Reconstruction.png'))

    else:
        fig, ax1 = plt.subplots(1, 1)
        fig.suptitle('Reconstruction of input sequence')
        ax1.plot(data_orig[1,...], color='k', label='Sequence Data')
        ax1.plot(data_tilde[1,...], color='r', linestyle='dashed', label='Sequence Reconstruction')

        fig.savefig(filepath+'evaluate/'+'Reconstruction_'+model_name+'.png')


def plot_loss(cfg, filepath, model_name):
    basepath = os.path.join(cfg['project_path'],"model","model_losses")
    train_loss = np.load(os.path.join(basepath,'train_losses_'+model_name+'.npy'))
    test_loss = np.load(os.path.join(basepath,'test_losses_'+model_name+'.npy'))
    mse_loss_train = np.load(os.path.join(basepath,'mse_train_losses_'+model_name+'.npy'))
    mse_loss_test = np.load(os.path.join(basepath,'mse_test_losses_'+model_name+'.npy'))
    km_loss = np.load(os.path.join(basepath,'kmeans_losses_'+model_name+'.npy'), allow_pickle=True)
    kl_loss = np.load(os.path.join(basepath,'kl_losses_'+model_name+'.npy'))
    fut_loss = np.load(os.path.join(basepath,'fut_losses_'+model_name+'.npy'))

    km_losses = []
    for i in range(len(km_loss)):
        km = km_loss[i].cpu().detach().numpy()
        km_losses.append(km)

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle('Losses of our Model')
    ax1.set(xlabel='Epochs', ylabel='loss [log-scale]')
    ax1.set_yscale("log")
    ax1.plot(train_loss, label='Train-Loss')
    ax1.plot(test_loss, label='Test-Loss')
    ax1.plot(mse_loss_train, label='MSE-Train-Loss')
    ax1.plot(mse_loss_test, label='MSE-Test-Loss')
    ax1.plot(km_losses, label='KMeans-Loss')
    ax1.plot(kl_loss, label='KL-Loss')
    ax1.plot(fut_loss, label='Prediction-Loss')
    ax1.legend()
    #fig.savefig(filepath+'evaluate/'+'MSE-and-KL-Loss'+model_name+'.png')
    fig.savefig(os.path.join(filepath,"evaluate",'MSE-and-KL-Loss'+model_name+'.png'))


def eval_temporal(cfg, use_gpu, model_name):

    SEED = 19
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']
    NUM_FEATURES = cfg['num_features']
    TEST_BATCH_SIZE = 64
    PROJECT_PATH = cfg['project_path']
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']

    filepath = os.path.join(cfg['project_path'],"model")


    seq_len_half = int(TEMPORAL_WINDOW/2)
    if use_gpu:
        torch.cuda.manual_seed(SEED)
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                        dropout_rec, dropout_pred).cuda()
        model.load_state_dict(torch.load(os.path.join(cfg['project_path'],"model","best_model",model_name+'_'+cfg['Project']+'.pkl')))
    else:
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                        dropout_rec, dropout_pred).to()

        model.load_state_dict(torch.load(os.path.join(cfg['project_path'],"model","best_model",model_name+'_'+cfg['Project']+'.pkl'), map_location=torch.device('cpu')))

    model.eval() #toggle evaluation mode

    testset = SEQUENCE_DATASET(os.path.join(cfg['project_path'],"data", "train",""), data='test_seq.npy', train=False, temporal_window=TEMPORAL_WINDOW)
    test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)

    plot_reconstruction(filepath, test_loader, seq_len_half, model, model_name, FUTURE_DECODER, FUTURE_STEPS)
    if use_gpu:
        plot_loss(cfg, filepath, model_name)
    else:
        pass #note, loading of losses needs to be adapted for CPU use #TODO



def evaluate_model(config, model_name):
    """
        Evaluation of testset
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)

    if not os.path.exists(os.path.join(cfg['project_path'],"model","evaluate")):
        os.mkdir(os.path.join(cfg['project_path'],"model","evaluate"))

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
        print('GPU active:',torch.cuda.is_available())
        print('GPU used:',torch.cuda.get_device_name(0))
    else:
        torch.device("cpu")
        print("CUDA is not working, or a GPU is not found; using CPU!")

    print("\n\nEvaluation of %s model. \n" %model_name)
    eval_temporal(cfg, use_gpu, model_name)

    print("You can find the results of the evaluation in '/Your-VAME-Project-Apr30-2020/model/evaluate/' \n"
          "OPTIONS:\n"
          "- vame.behavior_segmentation() to identify behavioral motifs.\n"
          "- re-run the model for further fine tuning. Check again with vame.evaluate_model()")
