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
import tqdm
import h5py
import numpy as np
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from vame.util.auxiliary import read_config
from vame.util.gif_pose_helper import get_animal_frames


def create_video(path_to_file, file, embed, clabel, frames, start, length): 
    # set matplotlib colormap
    cmap = matplotlib.cm.gray
    cmap_reversed = matplotlib.cm.get_cmap('gray_r')
    
    # this here generates every frame for your gif. The gif is lastly created by using ImageJ
    # the embed variable is my umap embedding, which is for the 2D case a 2xn dimensional vector
    fig = plt.figure()
    spec = GridSpec(ncols=2, nrows=1, width_ratios=[6, 3])
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1])
    ax2.axis('off')
    ax2.grid(False)
    lag = 0
    for i in tqdm.tqdm(range(length)):
        if i > 30:
            lag = i - 30
        ax1.cla()
        ax1.axis('off')
        ax1.grid(False)
        ax1.scatter(embed[:30000,0], embed[:30000,1], c=clabel[:30000], cmap='Spectral', s=1, alpha=0.4)
        ax1.set_aspect('equal', 'datalim')
        ax1.plot(embed[start+lag:start+i,0], embed[start+lag:start+i,1],'.b-',alpha=.6, linewidth=2, markersize=4)
        ax1.plot(embed[start+i,0], embed[start+i,1], 'gx', markersize=4)
        frame = frames[i]
        # lbl = label[start+i]
        ax2.imshow(frame, cmap=cmap_reversed)
        # ax2.set_title("Motif %d,\n Community: %s" % (lbl, motifs[lbl]), fontsize=10)
        fig.savefig(os.path.join(path_to_file,"gif_frames",file+'gif_%d.png') %i) 


def gif(config, pose_ref_index, start=None, length=500, file_format='.mp4', crop_size=(300,300)):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    
    files = []
    if cfg['all_data'] == 'No':
        all_flag = input("Do you want to write motif videos for your entire dataset? \n"
                     "If you only want to use a specific dataset type filename: \n"
                     "yes/no/filename ")
    else:
        all_flag = 'yes'

    if all_flag == 'yes' or all_flag == 'Yes':
        for file in cfg['video_sets']:
            files.append(file)

    elif all_flag == 'no' or all_flag == 'No':
        for file in cfg['video_sets']:
            use_file = input("Do you want to quantify " + file + "? yes/no: ")
            if use_file == 'yes':
                files.append(file)
            if use_file == 'no':
                continue
    else:
        files.append(all_flag)
        

    for file in files:
        path_to_file=os.path.join(cfg['project_path'],"results",file,"",model_name,"",'kmeans-'+str(n_cluster))
        if not os.path.exists(os.path.join(path_to_file,"gif_frames")):
            os.mkdir(os.path.join(path_to_file,"gif_frames"))
        
        embed = np.load(os.path.join(path_to_file,"community","","umap_embedding_"+file+'.npy'))
        community_label = np.load(os.path.join(path_to_file,"community","","community_label_"+file+'.npy'))
        
        if start == None:
            start = np.random.choice(community_label[:30000].shape[0]-length)
        else:
            start = start
        
        frames = get_animal_frames(cfg, file, pose_ref_index, start, length, file_format, crop_size)
        create_video(path_to_file, file, embed, community_label, frames, start, length)
                   
        

























