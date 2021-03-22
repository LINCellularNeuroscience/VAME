#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import vame

# These paths have to be set manually 
working_directory = '/YOUR/WORKING/DIRECTORY/'
project='Your-VAME-Project'
videos = ['/directory/to/your/video-1','/directory/to/your/video-2','...']
    
# Initialize your project
# Step 1.1:
config = vame.init_new_project(project=project, videos=videos, working_directory=working_directory, videotype='.mp4')

# After the inital creation of your project you can always access the config.yaml file 
# via specifying the path to your project
config = '/YOUR/WORKING/DIRECTORY/Your-VAME-Project-Apr14-2020/config.yaml'

# Step 1.2:
# Align your behavior videos egocentric and create training dataset:
# pose_ref_index: list of reference coordinate indices for alignment
# Example: 0: snout, 1: forehand_left, 2: forehand_right, 3: hindleft, 4: hindright, 5: tail
vame.egocentric_alignment(config, pose_ref_index=[0,5])

# If your experiment is by design egocentrical (e.g. head-fixed experiment on treadmill etc) 
# you can use the following to convert your .csv to a .npy array, ready to train vame on it
vame.csv_to_numpy(config, datapath='C:\\Research\\VAME\\vame_alpha_release-Mar16-2021\\videos\\pose_estimation\\')

# Step 1.3:
# create the training set for the VAME model
vame.create_trainset(config)

# Step 2:
# Train VAME:
vame.train_model(config)

# Step 3:
# Evaluate model
vame.evaluate_model(config)

# Step 4:
# Segment motifs/pose
vame.pose_segmentation(config)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# The following are optional choices to create motif videos, communities/hierarchies of behavior,
# community videos

# OPTIONIAL: Create motif videos to get insights about the fine grained poses
vame.motif_videos(config, videoType='.mp4')

# OPTIONAL: Create behavioural hierarchies via community detection
vame.community(config, umap_vis=False, cut_tree=2)

# OPTIONAL: Create community videos to get insights about behavior on a hierarchical scale
vame.community_videos(config)

# OPTIONAL: Down projection of latent vectors and visualization via UMAP
vame.visualization(config, label=None) #options: label: None, "motif", "community"

# OPTIONAL: Use the generative model (reconstruction decoder) to sample from 
# the learned data distribution, reconstruct random real samples or visualize
# the cluster center for validation
vame.generative_model(config, mode="centers") #options: mode: "sampling", "reconstruction", "centers

# OPTIONAL: Create a video of an egocentrically aligned mouse + path through 
# the community space (similar to our gif on github) to learn more about your representation
# and have something cool to show around ;) 
# Note: This function is currently very slow. Once the frames are saved you can create a video
# or gif via e.g. ImageJ or other tools
vame.gif(config, pose_ref_index=[0,5], start=0, length=500, file_format='.mp4', crop_size=(300,300))





