#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os


class SEQUENCE_DATASET(Dataset):
    def __init__(self,path_to_file,data,train,temporal_window):
        self.temporal_window = temporal_window        
        self.X = np.load(path_to_file+data)
        if self.X.shape[0] > self.X.shape[1]:
            self.X=self.X.T
            
        self.data_points = len(self.X[0,:])
        
        if train and not os.path.exists(path_to_file+'seq_mean.npy'):
            print("Compute mean and std for temporal dataset.")
            self.mean = np.mean(self.X)
            self.std = np.std(self.X)
            np.save(path_to_file+'seq_mean.npy', self.mean)
            np.save(path_to_file+'seq_std.npy', self.std)
        else:
            self.mean = np.load(path_to_file+'seq_mean.npy')
            self.std = np.load(path_to_file+'seq_std.npy')
        
        if train:
            print('Initialize train data. Datapoints %d' %self.data_points)
        else:
            print('Initialize test data. Datapoints %d' %self.data_points)
        
    def __len__(self):        
        return self.data_points

    def __getitem__(self, index):
        temp_window = self.temporal_window
        
        nf = self.data_points
        start = np.random.choice(nf-temp_window) 
        end = start+temp_window
        
        sequence = self.X[:,start:end]  

        sequence = (sequence-self.mean)/self.std
            
        return torch.from_numpy(sequence)
    
    
    
    
    
