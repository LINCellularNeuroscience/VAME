#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:52:04 2020

@author: luxemk
"""

import numpy as np
import pandas as pd

def csv_to_numpy(datapath, outputpath):
    """
    This is a demo function to show how a conversion from the resulting pose-estimation.csv file
    to a numpy array can be implemented.
    Note that this code is only useful for data which is a priori egocentric, i.e. head-fixed
    or otherwise restrained animals.

    example:
    vame.csv_to_npy('your/CSVfile/path', 'path/you/want/it/togo')
    """

    # Read in your .csv file, skip the first two rows and create a numpy array
    data = pd.read_csv(datapath, skiprows = 2)
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
    np.save(outputpath+"-PE-seq.npy", final_positions)
    print("conversion from DeepLabCut csv to numpy complete...")
