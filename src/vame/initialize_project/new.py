#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0

The following code is adapted from:

DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
from pathlib import Path
import shutil
from datetime import datetime as dt
from vame.util import auxiliary
from typing import List


def init_new_project(
    project: str,
    videos: List[str],
    poses_estimations: List[str],
    working_directory: str = None,
    videotype: str = '.mp4'
) -> str:
    """Creates a new VAME project with the given parameters.

    Args:
        project (str): Project name.
        videos (List[str]): List of videos paths to be used in the project. E.g. ['./sample_data/Session001.mp4']
        poses_estimations (List[str]): List of pose estimation files paths to be used in the project. E.g. ['./sample_data/pose estimation/Session001.csv']        working_directory (str, optional): _description_. Defaults to None.
        videotype (str, optional): Video extension (.mp4 or .avi). Defaults to '.mp4'.

    Returns:
        projconfigfile (str): Path to the new vame project config file.
    """

    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    year = date.year
    d = str(month[0:3]+str(day))
    date = dt.today().strftime('%Y-%m-%d')

    if working_directory == None:
        working_directory = '.'

    wd = Path(working_directory).resolve()
    project_name = '{pn}-{date}'.format(pn=project, date=d+'-'+str(year))

    project_path = wd / project_name


    if project_path.exists():
        print('Project "{}" already exists!'.format(project_path))
        projconfigfile = os.path.join(str(project_path),'config.yaml')
        return projconfigfile

    video_path = project_path / 'videos'
    data_path = project_path / 'data'
    results_path = project_path / 'results'
    model_path = project_path / 'model'

    for p in [video_path, data_path, results_path, model_path]:
        p.mkdir(parents=True)
        print('Created "{}"'.format(p))

    vids = []
    for i in videos:
        #Check if it is a folder
        if os.path.isdir(i):
            vids_in_dir = [os.path.join(i,vp) for vp in os.listdir(i) if videotype in vp]
            vids = vids + vids_in_dir
            if len(vids_in_dir)==0:
                print("No videos found in",i)
                print("Perhaps change the videotype, which is currently set to:", videotype)
            else:
                videos = vids
                print(len(vids_in_dir)," videos from the directory" ,i, "were added to the project.")
        else:
            if os.path.isfile(i):
                vids = vids + [i]
            videos = vids


    videos = [Path(vp) for vp in videos]
    video_names = []
    dirs_data = [data_path/Path(i.stem) for i in videos]
    for p in dirs_data:
        """
        Creates directory under data
        """
        p.mkdir(parents = True, exist_ok = True)
        video_names.append(p.stem)

    dirs_results = [results_path/Path(i.stem) for i in videos]
    for p in dirs_results:
        """
        Creates directory under results
        """
        p.mkdir(parents = True, exist_ok = True)

    destinations = [video_path.joinpath(vp.name) for vp in videos]

    os.mkdir(str(project_path)+'/'+'videos/pose_estimation/')
    os.mkdir(str(project_path)+'/model/pretrained_model')

    print("Copying the videos \n")
    for src, dst in zip(videos, destinations):
        shutil.copy(os.fspath(src),os.fspath(dst))

    print("Copying pose estimation files\n")
    for src, dst in zip(poses_estimations, [str(project_path)+'/videos/pose_estimation/'+Path(p).name for p in poses_estimations]):
        print('Copying ',src,' to ',dst)
        shutil.copy(os.fspath(src),os.fspath(dst))


    cfg_file,ruamelFile = auxiliary.create_config_template()
    cfg_file

    cfg_file['Project']=str(project)
    cfg_file['project_path']=str(project_path)+'/'
    cfg_file['test_fraction']=0.1
    cfg_file['video_sets']=video_names
    cfg_file['all_data']='yes'
    cfg_file['load_data']='-PE-seq-clean'
    cfg_file['anneal_function']='linear'
    cfg_file['batch_size']=256
    cfg_file['max_epochs']=500
    cfg_file['transition_function']='GRU'
    cfg_file['beta']=1
    cfg_file['zdims']=30
    cfg_file['learning_rate']=5e-4
    cfg_file['time_window']=30
    cfg_file['prediction_decoder']=1
    cfg_file['prediction_steps']=15
    cfg_file['model_convergence']=50
    cfg_file['model_snapshot']=50
    cfg_file['num_features']=12
    cfg_file['savgol_filter']=True
    cfg_file['savgol_length']=5
    cfg_file['savgol_order']=2
    cfg_file['hidden_size_layer_1']=256
    cfg_file['hidden_size_layer_2']=256
    cfg_file['dropout_encoder']=0
    cfg_file['hidden_size_rec']=256
    cfg_file['dropout_rec']=0
    cfg_file['hidden_size_pred']=256
    cfg_file['dropout_pred']=0
    cfg_file['kl_start']=2
    cfg_file['annealtime']=4
    cfg_file['mse_reconstruction_reduction']='sum'
    cfg_file['mse_prediction_reduction']='sum'
    cfg_file['kmeans_loss']=cfg_file['zdims']
    cfg_file['kmeans_lambda']=0.1
    cfg_file['scheduler']=1
    cfg_file['length_of_motif_video'] = 1000
    cfg_file['noise'] = False
    cfg_file['scheduler_step_size'] = 100
    cfg_file['legacy'] = False
    cfg_file['individual_parametrization'] = False
    cfg_file['random_state_kmeans'] = 42
    cfg_file['n_init_kmeans'] = 15
    cfg_file['model_name']='VAME'
    cfg_file['n_cluster'] = 15
    cfg_file['pretrained_weights'] = False
    cfg_file['pretrained_model']='None'
    cfg_file['min_dist'] = 0.1
    cfg_file['n_neighbors'] = 200
    cfg_file['random_state'] = 42
    cfg_file['num_points'] = 30000
    cfg_file['scheduler_gamma'] = 0.2
    cfg_file['softplus'] = False
    cfg_file['pose_confidence'] = 0.99
    cfg_file['iqr_factor'] = 4
    cfg_file['robust'] = True
    cfg_file['beta_norm'] = False
    cfg_file['n_layers'] = 1
    cfg_file['axis'] = None
    cfg_file['parametrization'] = 'hmm'
    cfg_file['hmm_trained'] = False

    projconfigfile=os.path.join(str(project_path),'config.yaml')
    # Write dictionary to yaml  config file
    auxiliary.write_config(projconfigfile,cfg_file)

    print('A VAME project has been created. \n')
    print('Now its time to prepare your data for VAME. '
          'The first step is to move your pose .csv file (e.g. DeepLabCut .csv) into the '
          '//YOUR//VAME//PROJECT//videos//pose_estimation folder. From here you can call '
          'either the function vame.egocentric_alignment() or if your data is by design egocentric '
          'call vame.csv_to_numpy(). This will prepare the data in .csv into the right format to start '
          'working with VAME.')

    return projconfigfile
