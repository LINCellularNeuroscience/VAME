#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:50:23 2019

@author: luxemk
"""

import vame

# These paths have to be set manually 
working_directory = '/YOUR/WORKING/DIRECTORY/'
project='Your-VAME-Project'
videos = ['/directory/to/your/video-1','/directory/to/your/video-2','...']
    
# Initialize your project
# Step 1:
config = vame.init_new_project(project=project, videos=videos, working_directory=working_directory)

# After the inital creation of your project you can always access the config.yaml file 
# via specifying the path to your project
config = '/YOUR/WORKING/DIRECTORY/Your-VAME-Project-Apr14-2020/config.yaml'

# Align your behavior videos egocentric and create training dataset:
# Make sure to put them into the data folder for every video. The name of this file is the video name + -PE-seq.npy: 
# E.g.: /Your-VAME-Project/data/video-1/video-1-PE-seq.npy
vame.create_trainset(config)

# Step 2:
# Train rnn model:
vame.rnn_model(config, model_name='VAME', pretrained_weights=False, pretrained_model='pretrained')

# Step 3:
# Evaluate model
vame.evaluate_model(config, model_name='VAME')

# Step 4:
# Quantify Behavior
vame.behavior_segmentation(config, model_name='VAME', cluster_method='kmeans', n_cluster=[30])

# Step 5:
# Get behavioral transition matrix, model usage and graph
vame.behavior_quantification(config, model_name='VAME', cluster_method='kmeans', n_cluster=30)


