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
from vame.util.auxiliary import read_config


#Helper function to return indexes of nans        
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0] 

#Interpolates all nan values of given array
def interpol(arr):
        
    y = np.transpose(arr)
     
    nans, x = nan_helper(y[0])
    y[0][nans]= np.interp(x(nans), x(~nans), y[0][~nans])   
    nans, x = nan_helper(y[1])
    y[1][nans]= np.interp(x(nans), x(~nans), y[1][~nans])
    
    arr = np.transpose(y)
    
    return arr

def csv_to_numpy(config):
    """
    This is a function to convert your pose-estimation.csv file to a numpy array.

    Note that this code is only useful for data which is a priori egocentric, i.e. head-fixed
    or otherwise restrained animals.

    example use:
    vame.csv_to_npy('pathto/your/config/yaml', 'path/toYourFolderwithCSV/')
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)

    path_to_file = cfg['project_path']
    filename = cfg['video_sets']
    confidence = cfg['pose_confidence']
    if cfg['egocentric_data'] == False:
        raise ValueError("The config.yaml indicates that the data is not egocentric. Please check the parameter egocentric_data")

    for file in filename:
        print(file)
        # Read in your .csv file, skip the first two rows and create a numpy array
        data = pd.read_csv(os.path.join(path_to_file,"videos","pose_estimation",file+'.csv'), skiprows = 3, header=None)
        data_mat = pd.DataFrame.to_numpy(data)
        data_mat = data_mat[:,1:]

        pose_list = []
        
        # get the number of bodyparts, their x,y-position and the confidence from DeepLabCut
        for i in range(int(data_mat.shape[1]/3)):
            pose_list.append(data_mat[:,i*3:(i+1)*3])
         
        # find low confidence and set them to NaN
        for i in pose_list:
            for j in i:
                if j[2] <= confidence:
                    j[0],j[1] = np.nan, np.nan        
         
        # interpolate NaNs
        for i in pose_list:
             i = interpol(i)
            
        positions = np.concatenate(pose_list, axis=1)
        final_positions = np.zeros((data_mat.shape[0], int(data_mat.shape[1]/3)*2))
        
        jdx = 0
        idx = 0
        for i in range(int(data_mat.shape[1]/3)):
            final_positions[:,idx:idx+2] = positions[:,jdx:jdx+2]
            jdx += 3
            idx += 2

        # save the final_positions array with np.save()
        np.save(os.path.join(path_to_file,'data',file,file+"-PE-seq.npy"), final_positions.T)
        print("conversion from DeepLabCut csv to numpy complete...")

    print("Your data is now ine right format and you can call vame.create_trainset()")
