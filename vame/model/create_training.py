#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
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
from scipy.stats import iqr
import matplotlib.pyplot as plt

from vame.util.auxiliary import read_config


#Helper function to return indexes of nans
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

#Interpolates all nan values of given array
def interpol(arr):
    y = np.transpose(arr)
    nans, x = nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    arr = np.transpose(y)
    return arr

def plot_check_parameter(cfg, iqr_val, num_frames, X_true, X_med, anchor_1, anchor_2):
    plot_X_orig = np.concatenate(X_true, axis=0).T
    plot_X_med = X_med.copy()
    iqr_cutoff = cfg['iqr_factor']*iqr_val
    
    plt.figure()
    plt.plot(plot_X_orig.T)
    plt.axhline(y=iqr_cutoff, color='r', linestyle='--', label="IQR cutoff")
    plt.axhline(y=-iqr_cutoff, color='r', linestyle='--')
    plt.title("Full Signal z-scored")
    plt.legend()
    
    if num_frames > 1000:
        rnd = np.random.choice(num_frames)
        
        plt.figure()
        plt.plot(plot_X_med[:,rnd:rnd+1000].T)
        plt.axhline(y=iqr_cutoff, color='r', linestyle='--', label="IQR cutoff")
        plt.axhline(y=-iqr_cutoff, color='r', linestyle='--')
        plt.title("Filtered signal z-scored")
        plt.legend()
        
        plt.figure()
        plt.plot(plot_X_orig[:,rnd:rnd+1000].T)
        plt.axhline(y=iqr_cutoff, color='r', linestyle='--', label="IQR cutoff")
        plt.axhline(y=-iqr_cutoff, color='r', linestyle='--')
        plt.title("Original signal z-scored")
        plt.legend()
        
        plt.figure()
        plt.plot(plot_X_orig[:,rnd:rnd+1000].T, 'g', alpha=0.5)
        plt.plot(plot_X_med[:,rnd:rnd+1000].T, '--m', alpha=0.6)
        plt.axhline(y=iqr_cutoff, color='r', linestyle='--', label="IQR cutoff")
        plt.axhline(y=-iqr_cutoff, color='r', linestyle='--')
        plt.title("Overlayed z-scored")
        plt.legend()
        
        # plot_X_orig = np.delete(plot_X_orig.T, anchor_1, 1)
        # plot_X_orig = np.delete(plot_X_orig, anchor_2, 1)
        # mse = (np.square(plot_X_orig[rnd:rnd+1000, :] - plot_X_med[:,rnd:rnd+1000].T)).mean(axis=0)
        
        
    else:
        plt.figure()
        plt.plot(plot_X_med.T)
        plt.axhline(y=iqr_cutoff, color='r', linestyle='--', label="IQR cutoff")
        plt.axhline(y=-iqr_cutoff, color='r', linestyle='--')
        plt.title("Filtered signal z-scored")
        plt.legend()
        
        plt.figure()
        plt.plot(plot_X_orig.T)
        plt.axhline(y=iqr_cutoff, color='r', linestyle='--', label="IQR cutoff")
        plt.axhline(y=-iqr_cutoff, color='r', linestyle='--')
        plt.title("Original signal z-scored")
        plt.legend()
        
    print("Please run the function with check_parameter=False if you are happy with the results")

def traindata_aligned(cfg, files, testfraction, num_features, savgol_filter, check_parameter):
    
    X_train = []
    pos = []
    pos_temp = 0
    pos.append(0)
    
    if check_parameter == True:
        X_true = []
        files = [files[0]]
        
    for file in files:
        print("z-scoring of file %s" %file)
        path_to_file = os.path.join(cfg['project_path'],"data", file, file+'-PE-seq.npy')
        data = np.load(path_to_file)
        
        X_mean = np.mean(data,axis=None)
        X_std = np.std(data, axis=None)
        X_z = (data.T - X_mean) / X_std
        
        # Introducing artificial error spikes
        # rang = [1.5, 2, 2.5, 3, 3.5, 3, 3, 2.5, 2, 1.5]
        # for i in range(num_frames):
        #     if i % 300 == 0:
        #         rnd = np.random.choice(12,2)
        #         for j in range(10):
        #             X_z[i+j, rnd[0]] = X_z[i+j, rnd[0]] * rang[j]
        #             X_z[i+j, rnd[1]] = X_z[i+j, rnd[1]] * rang[j]
                
        if check_parameter == True:
            X_z_copy = X_z.copy()
            X_true.append(X_z_copy)
            
        if cfg['robust'] == True:
            iqr_val = iqr(X_z)
            print("IQR value: %.2f, IQR cutoff: %.2f" %(iqr_val, cfg['iqr_factor']*iqr_val))
            for i in range(X_z.shape[0]):
                for marker in range(X_z.shape[1]):
                    if X_z[i,marker] > cfg['iqr_factor']*iqr_val:
                        X_z[i,marker] = np.nan
                        
                    elif X_z[i,marker] < -cfg['iqr_factor']*iqr_val:
                        X_z[i,marker] = np.nan       

            X_z = interpol(X_z)
             
        X_len = len(data.T)
        pos_temp += X_len
        pos.append(pos_temp)
        X_train.append(X_z)
    
    X = np.concatenate(X_train, axis=0)
    # X_std = np.std(X)
    
    detect_anchors = np.std(X.T, axis=1)
    sort_anchors = np.sort(detect_anchors)
    if sort_anchors[0] == sort_anchors[1]:
        anchors = np.where(detect_anchors == sort_anchors[0])[0]
        anchor_1_temp = anchors[0]
        anchor_2_temp = anchors[1]
        
    else:
        anchor_1_temp = int(np.where(detect_anchors == sort_anchors[0])[0])
        anchor_2_temp = int(np.where(detect_anchors == sort_anchors[1])[0])
    
    if anchor_1_temp > anchor_2_temp:
        anchor_1 = anchor_1_temp
        anchor_2 = anchor_2_temp
        
    else:
        anchor_1 = anchor_2_temp
        anchor_2 = anchor_1_temp
    
    X = np.delete(X, anchor_1, 1)
    X = np.delete(X, anchor_2, 1)
    
    X = X.T
    
    if savgol_filter:
        X_med = scipy.signal.savgol_filter(X, cfg['savgol_length'], cfg['savgol_order'])
    else:
        X_med = X
        
    num_frames = len(X_med.T)
    test = int(num_frames*testfraction)
    
    z_test =X_med[:,:test]
    z_train = X_med[:,test:]
      
    if check_parameter == True:
        plot_check_parameter(cfg, iqr_val, num_frames, X_true, X_med, anchor_1, anchor_2)
        
    else:        
        #save numpy arrays the the test/train info:
        np.save(os.path.join(cfg['project_path'],"data", "train",'train_seq.npy'), z_train)
        np.save(os.path.join(cfg['project_path'],"data", "train", 'test_seq.npy'), z_test)
        
        for i, file in enumerate(files):
            np.save(os.path.join(cfg['project_path'],"data", file, file+'-PE-seq-clean.npy'), X_med[:,pos[i]:pos[i+1]])
        
        print('Lenght of train data: %d' %len(z_train.T))
        print('Lenght of test data: %d' %len(z_test.T))
    

def traindata_fixed(cfg, files, testfraction, num_features, savgol_filter, check_parameter):
    X_train = []
    pos = []
    pos_temp = 0
    pos.append(0)
    
    if check_parameter == True:
        X_true = []
        rnd_file = np.random.choice(len(files))
        files = [files[0]]
        
    for file in files:
        print("z-scoring of file %s" %file)
        path_to_file = os.path.join(cfg['project_path'],"data", file, file+'-PE-seq.npy')
        data = np.load(path_to_file)
        X_mean = np.mean(data,axis=None)
        X_std = np.std(data, axis=None)
        X_z = (data.T - X_mean) / X_std
        
        if check_parameter == True:
            X_z_copy = X_z.copy()
            X_true.append(X_z_copy)
        
        if cfg['robust'] == True:
            iqr_val = iqr(X_z)
            print("IQR value: %.2f, IQR cutoff: %.2f" %(iqr_val, cfg['iqr_factor']*iqr_val))
            for i in range(X_z.shape[0]):
                for marker in range(X_z.shape[1]):
                    if X_z[i,marker] > cfg['iqr_factor']*iqr_val:
                        X_z[i,marker] = np.nan
                        
                    elif X_z[i,marker] < -cfg['iqr_factor']*iqr_val:
                        X_z[i,marker] = np.nan       

                X_z[i,:] = interpol(X_z[i,:])      
        
        X_len = len(data.T)
        pos_temp += X_len
        pos.append(pos_temp)
        X_train.append(X_z)
    
    X = np.concatenate(X_train, axis=0).T

    if savgol_filter:
        X_med = scipy.signal.savgol_filter(X, cfg['savgol_length'], cfg['savgol_order'])   
    else:
        X_med = X
        
    num_frames = len(X_med.T)
    test = int(num_frames*testfraction)
    
    z_test =X_med[:,:test]
    z_train = X_med[:,test:]
    
    if check_parameter == True:
        plot_check_parameter(cfg, iqr_val, num_frames, X_true, X_med)
        
    else:
        #save numpy arrays the the test/train info:
        np.save(os.path.join(cfg['project_path'],"data", "train",'train_seq.npy'), z_train)
        np.save(os.path.join(cfg['project_path'],"data", "train", 'test_seq.npy'), z_test)
        
        for i, file in enumerate(files):
            np.save(os.path.join(cfg['project_path'],"data", file, file+'-PE-seq-clean.npy'), X_med[:,pos[i]:pos[i+1]])
        
        print('Lenght of train data: %d' %len(z_train.T))
        print('Lenght of test data: %d' %len(z_test.T))


def create_trainset(config, check_parameter=False):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    legacy = cfg['legacy']
    fixed = cfg['egocentric_data']
    
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
    if cfg['robust'] == True:
        print("Using robust setting to eliminate outliers! IQR factor: %d" %cfg['iqr_factor'])
        
    if fixed == False:
        traindata_aligned(cfg, files, cfg['test_fraction'], cfg['num_features'], cfg['savgol_filter'], check_parameter)
    else:
        traindata_fixed(cfg, files, cfg['test_fraction'], cfg['num_features'], cfg['savgol_filter'], check_parameter)
    
    if check_parameter == False:
        print("A training and test set has been created. Next step: vame.train_model()")
