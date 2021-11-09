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

def csv_to_numpy(config, datapath):
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

    for file in filename:
        # Read in your .csv file, skip the first two rows and create a numpy array
        data = pd.read_csv(f'{datapath}/{file}.csv', skiprows = 2)
        data_mat = pd.DataFrame.to_numpy(data)
        data_mat = data_mat[:,1:]

        # get the number of bodyparts, their x,y-position and the confidence from DeepLabCut
        bodyparts = int(np.size(data_mat[0,:]) / 3)
        positions = []
        confidence = []
        idx = 0
        for i in range(bodyparts):
            positions.append(data_mat[:,idx:idx+2])
            confidence.append(data_mat[:,idx+2])
            idx += 3

        body_position = np.concatenate(positions, axis=1)
        con_arr = np.array(confidence)

        # find low confidence and set them to NaN (vame.create_trainset(config) will interpolate these NaNs)
        body_position_nan = []
        idx = -1
        for i in range(bodyparts*2):
            if i % 2 == 0:
                idx +=1
            seq = body_position[:,i]
            seq[con_arr[idx,:]<.99] = np.NaN
            body_position_nan.append(seq)

        final_positions = np.array(body_position_nan)

        # save the final_positions array with np.save()
        np.save(os.path.join(path_to_file,'data',file,file+"-PE-seq.npy"), final_positions)
        print("conversion from DeepLabCut csv to numpy complete...")

    print("Your data is now ine right format and you can call vame.create_trainset()")
