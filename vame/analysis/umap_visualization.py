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
import umap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from vame.util.auxiliary import read_config


def umap_vis(file, embed):        
    fig = plt.figure(1)
    plt.scatter(embed[:,0], embed[:,1], s=2, alpha=.5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    

def umap_label_vis(file, embed, label, n_cluster):
    fig = plt.figure(1)
    plt.scatter(embed[:,0], embed[:,1],  c=label[:30000], cmap='Spectral', s=2, alpha=.7)
    plt.colorbar(boundaries=np.arange(n_cluster+1)-0.5).set_ticks(np.arange(n_cluster))
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)


def umap_vis_comm(file, embed, community_label):
    num = np.unique(community_label).shape[0]
    fig = plt.figure(1)
    plt.scatter(embed[:,0], embed[:,1],  c=community_label[:30000], cmap='Spectral', s=2, alpha=.7)
    plt.colorbar(boundaries=np.arange(num+1)-0.5).set_ticks(np.arange(num))
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    

def visualization(config, label=None):
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

    for idx, file in enumerate(files):
        path_to_file=os.path.join(cfg['project_path'],"results",file,"",model_name,"",'kmeans-'+str(n_cluster))
        
        try:
            embed = np.load(os.path.join(path_to_file,"","community","","umap_embedding_"+file+".npy"))
        except:
            print("Compute embedding for file %s" %file)
            reducer = umap.UMAP(n_components=2, min_dist=cfg['min_dist'], n_neighbors=cfg['n_neighbors'], 
                    random_state=cfg['random_state']) 
            
            latent_vector = np.load(os.path.join(path_to_file,"",'latent_vector_'+file+'.npy'))
            
            num_points = cfg['num_points']
            if num_points > latent_vector.shape[0]:
                num_points = latent_vector.shape[0]
            print("Embedding %d data points.." %num_points)
            
            embed = reducer.fit_transform(latent_vector[:num_points,:])
            np.save(os.path.join(path_to_file,"community","umap_embedding_"+file+'.npy'), embed)
            
        if label == None:                    
            umap_vis(file, embed)
            
        if label == 'motif':
            label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_km_label_'+file+'.npy'))
            umap_label_vis(file, embed, label, n_cluster)

        if label == "community":
            community_label = np.load(os.path.join(path_to_file,"","community","","community_label_"+file+".npy"))
            umap_vis_comm(file, embed, community_label)                                    











