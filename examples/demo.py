#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:50:23 2019

@author: luxemk
"""

import sys
sys.path.append('./VAME')
from VAME import vame

# These paths have to be set manually 
working_directory = '/home/luxemk/Research/'
project='Your-VAME-Project'
videos = ['/directory/to/your/video-1','/directory/to/your/video-2','...']
videos = ['/home/luxemk/Research/Data/VAME/OFA/control/7/mouse-3-1.mp4']
    
# Initialize your project
# Pose Estimation file has to be put manually into project folder "/VAME-Project/videos/pose-estimation/"
# Make sure the pose estimation files have the same name as the videos with an additional PE at the end
# example: video-1-PE.csv

# Step 1:
config = vame.init_new_project(project=project, videos=videos, working_directory=working_directory)

# After inital creation of your project you can always access the config.yaml file 
# via specifying the path to your project
config = '/home/luxemk/Research/Your-VAME-Project-Apr14-2020/config.yaml'

# Align behavior video egocentrically and create training dataset:
# Note: vame.align() is currently only applicable if your data is similar to our demo data.
# If this is not the case please make sure to align your data egocentrically and put them into the
# data folder for every video. The name of this file is the video name + -PE-seq.npy: 
# /Your-VAME-Project/data/video-1/video-1-PE-seq.npy
vame.align(config)
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


