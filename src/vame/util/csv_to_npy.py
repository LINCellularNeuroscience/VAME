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
from vame.schemas.states import PoseToNumpyFunctionSchema, save_state
from vame.logging.logger import VameLogger
from vame.util.data_manipulation import interpol_first_rows_nans, read_pose_estimation_file


logger_config = VameLogger(__name__)
logger = logger_config.logger



@save_state(model=PoseToNumpyFunctionSchema)
def pose_to_numpy(config: str, save_logs=False) -> None:
    """Converts a pose-estimation.csv file to a numpy array. Note that this code is only useful for data which is a priori egocentric, i.e. head-fixed
    or otherwise restrained animals.

    Raises:
        ValueError: If the config.yaml file indicates that the data is not egocentric.
    """
    try:
        config_file = Path(config).resolve()
        cfg = read_config(config_file)

        if save_logs:
            log_path = Path(cfg['project_path']) / 'logs' / 'pose_to_numpy.log'
            logger_config.add_file_handler(log_path)


        path_to_file = cfg['project_path']
        filename = cfg['video_sets']
        confidence = cfg['pose_confidence']
        if not cfg['egocentric_data']:
            raise ValueError("The config.yaml indicates that the data is not egocentric. Please check the parameter egocentric_data")

        folder_path = os.path.join(path_to_file,'videos','pose_estimation')
        for file in filename:
            data, data_mat = read_pose_estimation_file(
                folder_path=folder_path,
                filename=file,
                filetype=cfg['pose_estimation_filetype'],
                path_to_pose_nwb_series_data=cfg['path_to_pose_nwb_series_data']
            )

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
                i = interpol_first_rows_nans(i)

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
            logger.info("conversion from DeepLabCut csv to numpy complete...")

        logger.info("Your data is now in right format and you can call vame.create_trainset()")
    except Exception as e:
        logger.exception(f"{e}")
        raise e
    finally:
        logger_config.remove_file_handler()
