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

import os, yaml
from pathlib import Path
import ruamel.yaml


def create_config_template():
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.
    """
    import ruamel.yaml
    yaml_str = """\
# Project configurations
    Project:
    model_name:
    n_cluster:
    pose_confidence: 
    \n
# Project path and videos
    project_path:
    video_sets:
    \n
# Data
    all_data:
    \n
# Creation of train set:
    robust:
    iqr_factor:
    axis: 
    savgol_filter:
    savgol_length:
    savgol_order:
    test_fraction:
    \n
# RNN model general hyperparameter:
    pretrained_model: 
    pretrained_weights: 
    num_features:
    batch_size:
    max_epochs:
    model_snapshot:
    model_convergence:
    transition_function:
    beta:
    beta_norm:
    zdims:
    learning_rate:
    time_window:
    prediction_decoder:
    prediction_steps:
    noise: 
    scheduler:
    scheduler_step_size:
    scheduler_gamma:
    softplus: 
    \n
# Segmentation:
    load_data:
    individual_parameterization: 
    random_state_kmeans: 
    n_init_kmeans:
    \n
# Video writer:
    length_of_motif_video:
    \n
# UMAP parameter:
    min_dist:
    n_neighbors: 
    random_state: 
    num_points:
    \n
# ONLY CHANGE ANYTHING BELOW IF YOU ARE FAMILIAR WITH RNN MODELS
# RNN encoder hyperparamter:
    hidden_size_layer_1:
    hidden_size_layer_2:
    dropout_encoder:
    \n
# RNN reconstruction hyperparameter:
    hidden_size_rec:
    dropout_rec:
    n_layers:
    \n
# RNN prediction hyperparamter:
    hidden_size_pred:
    dropout_pred:
    \n
# RNN loss hyperparameter:
    mse_reconstruction_reduction:
    mse_prediction_reduction:
    kmeans_loss:
    kmeans_lambda:
    anneal_function:
    kl_start:
    annealtime:
    \n
# Legacy mode
    legacy: 
    """
    ruamelFile = ruamel.yaml.YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return(cfg_file,ruamelFile)


def read_config(configname):
    """
    Reads structured config file defining a project.
    """
    ruamelFile = ruamel.yaml.YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = ruamelFile.load(f)
                curr_dir = os.path.dirname(configname)
                if cfg["project_path"] != curr_dir:
                    cfg["project_path"] = curr_dir
                    write_config(configname, cfg)
        except Exception as err:
            if len(err.args) > 2:
                if (
                    err.args[2]
                    == "could not determine a constructor for the tag '!!python/tuple'"
                ):
                    with open(path, "r") as ymlfile:
                        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                        write_config(configname, cfg)
                else:
                    raise

    else:
        raise FileNotFoundError(
            "Config file is not found. Please make sure that the file exists and/or that you passed the path of the config file correctly!"
        )
    return cfg

def write_config(configname,cfg):
    """
    Write structured config file.
    """
    with open(configname, 'w') as cf:
        ruamelFile = ruamel.yaml.YAML()
        cfg_file,ruamelFile = create_config_template()
        for key in cfg.keys():
            cfg_file[key]=cfg[key]

        ruamelFile.dump(cfg_file, cf)

def update_config(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    project = cfg['Project']
    project_path = cfg['project_path']
    video_names = []
    for file in cfg['video_sets']:
        video_names.append(file)
    
    cfg_file,ruamelFile = create_config_template()
    
    cfg_file['Project']=str(project)
    cfg_file['project_path']=str(project_path)+'/'
    cfg_file['test_fraction']=.1
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
    cfg_file['individual_parameterization'] = False
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
    cfg_file['axis'] = 'None'
    
    projconfigfile=os.path.join(str(project_path),'config.yaml')
    # Write dictionary to yaml  config file
    write_config(projconfigfile,cfg_file)