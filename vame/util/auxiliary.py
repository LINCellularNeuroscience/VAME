#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 12:28:35 2019

@author: luxemk
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
# Project name
    Project:
    \n
# Project path and videos
    project_path:
    video_sets:
    \n
# Data
    all_data:
    \n
# Segmantation:
    load_data:
    snapshot:
    snapshot_epoch:
    median_filter:
    \n
# Creation of train set:
    savgol_filter:
    savgol_length:
    savgol_order:
    test_fraction:
    \n
# RNN model general hyperparameter:
    num_features:
    batch_size:
    epochs:
    model_snapshot:
    model_convergence:
    transition_function:
    beta:
    zdims:
    learning_rate:
    time_window: 
    prediction_decoder:
    prediction_steps:
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
    scheduler:
    """
    ruamelFile = ruamel.yaml.YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return(cfg_file,ruamelFile)


def read_config(configname):
    """
    Reads structured config file
    """
    ruamelFile = ruamel.yaml.YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                cfg = ruamelFile.load(f)
        except Exception as err:
            if err.args[2] == "could not determine a constructor for the tag '!!python/tuple'":
                with open(path, 'r') as ymlfile:
                  cfg = yaml.load(ymlfile,Loader=yaml.SafeLoader)
                  write_config(configname,cfg)
    else:
        raise FileNotFoundError ("Config file is not found. Please make sure that the file exists and/or there are no unnecessary spaces in the path of the config file!")
    return(cfg)
    
    
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
        
        
        
        
        
        
        