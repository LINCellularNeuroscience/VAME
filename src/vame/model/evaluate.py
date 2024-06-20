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
from typing import Optional

from vame.util.auxiliary import read_config
from vame.model.rnn_vae import RNN_VAE
from vame.model.dataloader import SEQUENCE_DATASET
from vame.logging.logger import VameLogger

logger_config = VameLogger(__name__)
logger = logger_config.logger



use_gpu = torch.cuda.is_available()
if use_gpu:
    pass
else:
    torch.device("cpu")


def plot_reconstruction(
    filepath: str,
    test_loader: Data.DataLoader,
    seq_len_half: int,
    model: RNN_VAE,
    model_name: str,
    FUTURE_DECODER: bool,
    FUTURE_STEPS: int,
    suffix: Optional[str] = None
) -> None:
    """
    Plot the reconstruction and future prediction of the input sequence.

    Args:
        filepath (str): Path to save the plot.
        test_loader (Data.DataLoader): DataLoader for the test dataset.
        seq_len_half (int): Half of the temporal window size.
        model (RNN_VAE): Trained VAE model.
        model_name (str): Name of the model.
        FUTURE_DECODER (bool): Flag indicating whether the model has a future prediction decoder.
        FUTURE_STEPS (int): Number of future steps to predict.
        suffix (Optional[str], optional): Suffix for the saved plot filename. Defaults to None.
    """
    #x = test_loader.__iter__().next()
    dataiter = iter(test_loader)
    x = next(dataiter)
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
        fig, axs = plt.subplots(2, 5)
        fig.suptitle('Reconstruction [top] and future prediction [bottom] of input sequence')
        for i in range(5):
            axs[0,i].plot(data_orig[i,...], color='k', label='Sequence Data')
            axs[0,i].plot(data_tilde[i,...], color='r', linestyle='dashed', label='Sequence Reconstruction')

            axs[1,i].plot(fut_orig[i,...], color='k')
            axs[1,i].plot(fut[i,...], color='r', linestyle='dashed')
        axs[0,0].set(xlabel='time steps', ylabel='reconstruction')
        axs[1,0].set(xlabel='time steps', ylabel='predction')
        fig.savefig(os.path.join(filepath,"evaluate",'Future_Reconstruction.png'))

    else:
        fig, ax1 = plt.subplots(1, 5)
        for i in range(5):
            fig.suptitle('Reconstruction of input sequence')
            ax1[i].plot(data_orig[i,...], color='k', label='Sequence Data')
            ax1[i].plot(data_tilde[i,...], color='r', linestyle='dashed', label='Sequence Reconstruction')
        fig.set_tight_layout(True)
        if not suffix:
            fig.savefig(os.path.join(filepath,'evaluate','Reconstruction_'+model_name+'.png'), bbox_inches='tight')
        elif suffix:
            fig.savefig(os.path.join(filepath,'evaluate','Reconstruction_'+model_name+'_'+suffix+'.png'), bbox_inches='tight')


def plot_loss(cfg: dict, filepath: str, model_name: str) -> None:
    """
    Plot the losses of the trained model.

    Args:
        cfg (dict): Configuration dictionary.
        filepath (str): Path to save the plot.
        model_name (str): Name of the model.
    """
    basepath = os.path.join(cfg['project_path'],"model","model_losses")
    train_loss = np.load(os.path.join(basepath,'train_losses_'+model_name+'.npy'))
    test_loss = np.load(os.path.join(basepath,'test_losses_'+model_name+'.npy'))
    mse_loss_train = np.load(os.path.join(basepath,'mse_train_losses_'+model_name+'.npy'))
    mse_loss_test = np.load(os.path.join(basepath,'mse_test_losses_'+model_name+'.npy'))
    km_losses = np.load(os.path.join(basepath,'kmeans_losses_'+model_name+'.npy'))
    kl_loss = np.load(os.path.join(basepath,'kl_losses_'+model_name+'.npy'))
    fut_loss = np.load(os.path.join(basepath,'fut_losses_'+model_name+'.npy'))

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
    fig.savefig(os.path.join(filepath,"evaluate",'MSE-and-KL-Loss'+model_name+'.png'))


def eval_temporal(
    cfg: dict,
    use_gpu: bool,
    model_name: str,
    fixed: bool,
    snapshot: Optional[str] = None,
    suffix: Optional[str] = None
) -> None:
    """
    Evaluate the temporal aspects of the trained model.

    Args:
        cfg (dict): Configuration dictionary.
        use_gpu (bool): Flag indicating whether to use GPU for evaluation.
        model_name (str): Name of the model.
        fixed (bool): Flag indicating whether the data is fixed or not.
        snapshot (Optional[str], optional): Path to the model snapshot. Defaults to None.
        suffix (Optional[str], optional): Suffix for the saved plot filename. Defaults to None.
    """
    SEED = 19
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']
    NUM_FEATURES = cfg['num_features']
    if not fixed:
        NUM_FEATURES = NUM_FEATURES - 2
    TEST_BATCH_SIZE = 64
    PROJECT_PATH = cfg['project_path']
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    softplus = cfg['softplus']

    filepath = os.path.join(cfg['project_path'],"model")


    seq_len_half = int(TEMPORAL_WINDOW/2)
    if use_gpu:
        torch.cuda.manual_seed(SEED)
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                        dropout_rec, dropout_pred, softplus).cuda()
        model.load_state_dict(torch.load(os.path.join(cfg['project_path'],"model","best_model",model_name+'_'+cfg['Project']+'.pkl')))
    else:
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                        dropout_rec, dropout_pred, softplus).to()
        if not snapshot:
            model.load_state_dict(torch.load(os.path.join(cfg['project_path'],"model","best_model",model_name+'_'+cfg['Project']+'.pkl'), map_location=torch.device('cpu')))
        elif snapshot:
            model.load_state_dict(torch.load(snapshot), map_location=torch.device('cpu'))
    model.eval() #toggle evaluation mode

    testset = SEQUENCE_DATASET(
        os.path.join(cfg["project_path"], "data", "train", ""),
        data="test_seq.npy",
        train=False,
        temporal_window=TEMPORAL_WINDOW,
        logger_config=logger_config,
    )
    test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)

    if not snapshot:
        plot_reconstruction(filepath, test_loader, seq_len_half, model, model_name, FUTURE_DECODER, FUTURE_STEPS)#, suffix=suffix
    elif snapshot:
        plot_reconstruction(filepath, test_loader, seq_len_half, model, model_name, FUTURE_DECODER, FUTURE_STEPS, suffix=suffix)#,
    if use_gpu:
        plot_loss(cfg, filepath, model_name)
    else:
        plot_loss(cfg, filepath, model_name)
        # pass #note, loading of losses needs to be adapted for CPU use #TODO


def evaluate_model(config: str, use_snapshots: bool = False, save_logs: bool = False) -> None:
    """Evaluate the trained model.

    Args:
        config (str): Path to config file.
        use_snapshots (bool, optional): Whether to plot for all snapshots or only the best model. Defaults to False.
    """
    try:
        config_file = Path(config).resolve()
        cfg = read_config(config_file)
        if save_logs:
            log_path = Path(cfg['project_path']) / 'logs' / 'evaluate_model.log'
            logger_config.add_file_handler(log_path)

        model_name = cfg['model_name']
        fixed = cfg['egocentric_data']

        if not os.path.exists(os.path.join(cfg['project_path'],"model","evaluate")):
            os.mkdir(os.path.join(cfg['project_path'],"model","evaluate"))

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("Using CUDA")
            logger.info('GPU active: {}'.format(torch.cuda.is_available()))
            logger.info('GPU used: {}'.format(torch.cuda.get_device_name(0)))
        else:
            torch.device("cpu")
            logger.info("CUDA is not working, or a GPU is not found; using CPU!")

        logger.info("Evaluation of %s model. " %model_name)
        if not use_snapshots:
            eval_temporal(cfg, use_gpu, model_name, fixed)#suffix=suffix
        elif use_snapshots:
            snapshots=os.listdir(os.path.join(cfg['project_path'],'model','best_model','snapshots'))
            for snap in snapshots:
                fullpath = os.path.join(cfg['project_path'],"model","best_model","snapshots",snap)
                epoch=snap.split('_')[-1]
                eval_temporal(cfg, use_gpu, model_name, fixed, snapshot=fullpath, suffix='snapshot'+str(epoch))
                #eval_temporal(cfg, use_gpu, model_name, legacy=legacy, suffix='bestModel')

        logger.info("You can find the results of the evaluation in '/Your-VAME-Project-Apr30-2020/model/evaluate/' \n"
            "OPTIONS:\n"
            "- vame.pose_segmentation() to identify behavioral motifs.\n"
            "- re-run the model for further fine tuning. Check again with vame.evaluate_model()")
    except Exception as e:
        logger.exception(f"An error occurred during model evaluation: {e}")
    finally:
        logger_config.remove_file_handler()