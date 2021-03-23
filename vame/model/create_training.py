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
import pandas as pd
from pathlib import Path
import scipy.signal

from vame.util.auxiliary import read_config


def traindata(cfg, files, testfraction, num_features, savgol_filter):
    
    X_train = []
    pos = []
    pos_temp = 0
    pos.append(0)
    for file in files:
        path_to_file = os.path.join(cfg['project_path'],"data", file, file+'-PE-seq.npy')
        data = np.load(path_to_file)
        X_mean = np.mean(data,axis=1)
        X_std = np.std(data, axis=1)
        X_z = (data.T - X_mean) / X_std
        X_len = len(data.T)
        pos_temp += X_len
        pos.append(pos_temp)
        X_train.append(X_z)
    
    X = np.concatenate(X_train, axis=0)
    
    detect_anchors = np.std(data, axis=1)
    sort_anchors = np.sort(detect_anchors)
    anchor_1 = int(np.where(detect_anchors == sort_anchors[0])[0])
    anchor_2 = int(np.where(detect_anchors == sort_anchors[1])[0])
    
    X = np.delete(X, anchor_1, 1)
    X = np.delete(X, anchor_2-1, 1)
    
    X = X.T
    
    if savgol_filter:
        X_med = scipy.signal.savgol_filter(X, cfg['savgol_length'], cfg['savgol_order'])
        
    num_frames = len(X_med.T)
    test = int(num_frames*testfraction)
    
    z_test =X_med[:,:test]
    z_train = X_med[:,test:]
        
    #save numpy arrays the the test/train info:
    np.save(os.path.join(cfg['project_path'],"data", "train",'train_seq.npy'), z_train)
    np.save(os.path.join(cfg['project_path'],"data", "train", 'test_seq.npy'), z_test)
    
    for i, file in enumerate(files):
        np.save(os.path.join(cfg['project_path'],"data", file, file+'-PE-seq-clean.npy'), X_med[:,pos[i]:pos[i+1]])
    
    print('Lenght of train data: %d' %len(z_train.T))
    print('Lenght of test data: %d' %len(z_test.T))
    

def traindata_legacy(cfg, files, testfraction, num_features, savgol_filter):

    X_train = []
    pos = []
    pos_temp = 0
    pos.append(0)
    for file in files:
        path_to_file= os.path.join(cfg['project_path'],"data", file, file+'-PE-seq.npy')
        print(path_to_file)
        X = np.load(path_to_file)
        X_len = len(X.T)
        pos_temp += X_len
        pos.append(pos_temp)
        X_train.append(X)

    X = np.concatenate(X_train, axis=1)

    seq_inter = np.zeros((X.shape[0],X.shape[1]))
    for s in range(num_features):
        seq_temp = X[s,:]
        seq_pd = pd.Series(seq_temp)
        if np.isnan(seq_pd[0]):
            seq_pd[0] = next(x for x in seq_pd if not np.isnan(x))
        seq_pd_inter = seq_pd.interpolate(method="linear", order=None)
        seq_inter[s,:] = seq_pd_inter

    if savgol_filter:
        X_med = scipy.signal.savgol_filter(seq_inter, cfg['savgol_length'], cfg['savgol_order'])
    num_frames = len(X_med.T)
    test = int(num_frames*testfraction)

    z_test =X_med[:,:test]
    z_train = X_med[:,test:]

    #save numpy arrays the the test/train info:
    np.save(os.path.join(cfg['project_path'],"data", "train",'train_seq.npy'), z_train)
    np.save(os.path.join(cfg['project_path'],"data", "train", 'test_seq.npy'), z_test)


    for i, file in enumerate(files):
        np.save(os.path.join(cfg['project_path'],"data", file, file+'-PE-seq-clean.npy'), X_med[:,pos[i]:pos[i+1]])

    print('Length of train data: %d' %len(z_train.T))
    print('Length of test data: %d' %len(z_test.T))


def create_trainset(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    legacy = cfg['legacy']
    
    if not os.path.exists(os.path.join(cfg['project_path'],'data','train',"")):
        os.mkdir(os.path.join(cfg['project_path'],'data','train',""))

    files = []
    if cfg['all_data'] == 'No':
        for file in cfg['video_sets']:
            use_file = input("Do you want to train on " + file + "? yes/no: ")
            if use_file == 'yes':
                files.append(file)
            if use_file == 'no':
                continue
    else:
        for file in cfg['video_sets']:
            files.append(file)

    print("Creating training dataset...")
    if legacy == False:
        traindata(cfg, files, cfg['test_fraction'], cfg['num_features'], cfg['savgol_filter'])
    else:
        traindata_legacy(cfg, files, cfg['test_fraction'], cfg['num_features'], cfg['savgol_filter'])
        
    print("A training and test set has been created. Now everything is ready to train a variational autoencoder"
          "via vame.train_model() ...")