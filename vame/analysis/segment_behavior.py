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
import numpy as np
from pathlib import Path

import torch
import scipy.signal
from sklearn import mixture
from sklearn.cluster import KMeans

from vame.util.auxiliary import read_config
from vame.model.rnn_vae import RNN_VAE


def load_data(PROJECT_PATH, file, data):
    X = np.load(os.path.join(PROJECT_PATH,"data",file,"",file+data+'.npy'))
    mean = np.load(os.path.join(PROJECT_PATH,"data","train",'seq_mean.npy'))
    std = np.load(os.path.join(PROJECT_PATH,"data","train",'seq_std.npy'))
    X = (X-mean)/std
    return X


def kmeans_clustering(context, n_clusters):
    kmeans = KMeans(init='k-means++',n_clusters=n_clusters, random_state=42,n_init=15).fit(context)
    return kmeans.predict(context)


def gmm_clustering(context,n_components):
    GMM = mixture.GaussianMixture
    gmm = GMM(n_components=n_components,covariance_type='full').fit(context)
    return gmm.predict(context)


def behavior_segmentation(config, model_name=None, cluster_method='kmeans', n_cluster=[30]):

    config_file = Path(config).resolve()
    cfg = read_config(config_file)

    for folders in cfg['video_sets']:
        if not os.path.exists(os.path.join(cfg['project_path'],"results",folders,"",model_name)):
            os.mkdir(os.path.join(cfg['project_path'],"results",folders,"",model_name))

    files = []
    if cfg['all_data'] == 'No':
        all_flag = input("Do you want to qunatify your entire dataset? \n"
                         "If you only want to use a specific dataset type filename: \n"
                         "yes/no/filename ")
    else:
        all_flag = 'yes'

    if all_flag == 'yes' or all_flag == 'Yes':
        for file in cfg['video_sets']:
            files.append(file)
    elif all_flag == 'no' or all_flag == 'No':
        for file in cfg['video_sets']:
            use_file = input("Do you want to quantify " + file + "? yes/no: ")
            if use_file == 'yes':
                files.append(file)
            if use_file == 'no':
                continue
    else:
        files.append(all_flag)


    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
        print('GPU active:',torch.cuda.is_available())
        print('GPU used:',torch.cuda.get_device_name(0))
    else:
        print("CUDA is not working! Attempting to use the CPU...")
        torch.device("cpu")

    z, z_logger = temporal_quant(cfg, model_name, files, use_gpu)
    cluster_latent_space(cfg, files, z, z_logger, cluster_method, n_cluster, model_name)


def temporal_quant(cfg, model_name, files, use_gpu):

    SEED = 19
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']
    NUM_FEATURES = cfg['num_features']
    PROJECT_PATH = cfg['project_path']
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    temp_win = int(TEMPORAL_WINDOW/2)

    if use_gpu:
        torch.cuda.manual_seed(SEED)
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                        dropout_rec, dropout_pred).cuda()
    else:
        torch.cuda.manual_seed(SEED)
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                        dropout_rec, dropout_pred).to()

    if cfg['snapshot'] == 'yes':
        if use_gpu:
            model.load_state_dict(torch.load(os.path.join(cfg['project_path'],"model","best_model","snapshots",model_name+'_'+cfg['Project']+'_epoch_'+cfg['snapshot_epoch']+'.pkl')))
        else:
            model.load_state_dict(torch.load(os.path.join(cfg['project_path'],"model","best_model","snapshots",model_name+'_'+cfg['Project']+'_epoch_'+cfg['snapshot_epoch']+'.pkl'),map_location=torch.device('cpu')))
    else:
        if use_gpu:
            model.load_state_dict(torch.load(os.path.join(cfg['project_path'],"model","best_model",model_name+'_'+cfg['Project']+'.pkl')))
        else:
            model.load_state_dict(torch.load(os.path.join(cfg['project_path'],"model","best_model",model_name+'_'+cfg['Project']+'.pkl'),map_location=torch.device('cpu')))

    model.eval()

    z_list = []
    z_logger = []
    logger = 0
    for file in files:
        print("Computing latent space for %s " %file)
        z_logger.append(logger)


        data=cfg['load_data']
        X = load_data(PROJECT_PATH, file, data)

        if X.shape[0] > X.shape[1]:
                X=X.T

        num_frames = len(X[0,:]) - temp_win
        window_start = int(temp_win/2)
        idx = int(temp_win/2)
        x_decoded = []

        with torch.no_grad():
            for i in range(num_frames):
                if idx >= num_frames:
                    break
                data = X[:,idx-window_start:idx+window_start]
                data = np.reshape(data, (1,temp_win,NUM_FEATURES))
                if use_gpu:
                    dataTorch = torch.from_numpy(data).type(torch.FloatTensor).cuda()
                else:
                    dataTorch = torch.from_numpy(data).type(torch.FloatTensor).to()
                h_n = model.encoder(dataTorch)
                latent, _, _ = model.lmbda(h_n)
                z = latent.cpu().data.numpy()
                x_decoded.append(z)
                idx += 1

        z_temp = np.concatenate(x_decoded,axis=0)
        logger_temp = len(z_temp)
        logger += logger_temp
        z_list.append(z_temp)

    z_array= np.concatenate(z_list)
    z_logger.append(len(z_array))

    return z_array, z_logger


def cluster_latent_space(cfg, files, z_data, z_logger, cluster_method, n_cluster, model_name):

    for cluster in n_cluster:
        if cluster_method == 'kmeans':
            print('Behavior segmentation via k-Means for %d cluster.' %cluster)
            data_labels = kmeans_clustering(z_data, n_clusters=cluster)
            data_labels = np.int64(scipy.signal.medfilt(data_labels, cfg['median_filter']))

        if cluster_method == 'GMM':
            print('Behavior segmentation via GMM.')
            data_labels = gmm_clustering(z_data, n_components=cluster)
            data_labels = np.int64(scipy.signal.medfilt(data_labels, cfg['median_filter']))

        for idx, file in enumerate(files):
            print("Segmentation for file %s..." %file )
            if not os.path.exists(os.path.join(cfg['project_path'],"results",file,"",model_name,"",cluster_method+'-'+str(cluster))):
                os.mkdir(os.path.join(cfg['project_path'],"results",file,"",model_name,"",cluster_method+'-'+str(cluster)))

            save_data = os.path.join(cfg['project_path'],"results",file,"",model_name,"")
            z_latent = z_data[z_logger[idx]:z_logger[idx+1],:]
            labels = data_labels[z_logger[idx]:z_logger[idx+1]]


            if cluster_method == 'kmeans':
                np.save(save_data+cluster_method+'-'+str(cluster)+'/'+str(cluster)+'_km_label_'+file, labels)
                np.save(save_data+cluster_method+'-'+str(cluster)+'/'+'latent_vector_'+file, z_latent)

            if cluster_method == 'GMM':
                np.save(save_data+cluster_method+'-'+str(cluster)+'/'+str(cluster)+'_gmm_label_'+file, labels)
                np.save(save_data+cluster_method+'-'+str(cluster)+'/'+'latent_vector_'+file, z_latent)

            if cluster_method == 'all':
                np.save(save_data+cluster_method+'-'+str(cluster)+'/'+str(cluster)+'_km_label_'+file, labels)
                np.save(save_data+cluster_method+'-'+str(cluster)+'/'+str(cluster)+'_gmm_label_'+file, labels)
                np.save(save_data+cluster_method+'-'+str(cluster)+'/'+'latent_vector_'+file, z_latent)
