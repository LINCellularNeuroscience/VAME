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
from typing import Tuple


def create_config_template() -> Tuple[dict, ruamel.yaml.YAML]:
    """
    Creates a template for the config.yaml file.

    Returns:
        Tuple[dict, ruamel.yaml.YAML]: A tuple containing the template dictionary and the Ruamel YAML instance.
    """
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
    egocentric_data:
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
    #Note the optimal scheduler threshold below can vary greatly (from .1-.0001) between experiments.
    #You are encouraged to read the torch.optim.ReduceLROnPlateau docs to understand the threshold to use.
    scheduler_threshold:
    softplus:
    \n
# Segmentation:
    parametrization:
    hmm_trained: False
    load_data:
    individual_parametrization:
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
    """
    ruamelFile = ruamel.yaml.YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return(cfg_file,ruamelFile)


def read_config(configname: str) -> dict:
    """
    Reads structured config file defining a project.

    Args:
        configname (str): Path to the config file.

    Returns:
        dict: The contents of the config file as a dictionary.
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


def write_config(configname: str, cfg: dict) -> None:
    """
    Write structured config file.

    Args:
        configname (str): Path to the config file.
        cfg (dict): Dictionary containing the config data.
    """
    with open(configname, 'w') as cf:
        ruamelFile = ruamel.yaml.YAML()
        cfg_file,ruamelFile = create_config_template()
        for key in cfg.keys():
            cfg_file[key]=cfg[key]

        ruamelFile.dump(cfg_file, cf)
