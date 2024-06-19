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
from typing import Optional, Union
from vame.util.auxiliary import read_config


def umap_vis(file: str, embed: np.ndarray, num_points: int) -> None:
    """
    Visualize UMAP embedding without labels.

    Args:
        file (str): Name of the file (deprecated).
        embed (np.ndarray): UMAP embedding.
        num_points (int): Number of data points to visualize.

    Returns:
        None - Plot Visualization of UMAP embedding.
    """
    plt.cla()
    plt.clf()
    plt.close('all')
    fig = plt.figure(1)
    plt.scatter(embed[:num_points,0], embed[:num_points,1], s=2, alpha=.5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    return fig


def umap_label_vis(file: str, embed: np.ndarray, label: np.ndarray, n_cluster: int, num_points: int) -> None:
    """
    Visualize UMAP embedding with motif labels.

    Args:
        file (str): Name of the file (deprecated).
        embed (np.ndarray): UMAP embedding.
        label (np.ndarray): Motif labels.
        n_cluster (int): Number of clusters.
        num_points (int): Number of data points to visualize.

    Returns:
        fig - Plot figure of UMAP visualization embedding with motif labels.
    """
    fig = plt.figure(1)
    plt.scatter(embed[:num_points,0], embed[:num_points,1],  c=label[:num_points], cmap='Spectral', s=2, alpha=.7)
    #plt.colorbar(boundaries=np.arange(n_cluster+1)-0.5).set_ticks(np.arange(n_cluster))
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    return fig


def umap_vis_comm(file: str, embed: np.ndarray, community_label: np.ndarray, num_points: int) -> None:
    """
    Visualize UMAP embedding with community labels.

    Args:
        file (str): Name of the file (deprecated).
        embed (np.ndarray): UMAP embedding.
        community_label (np.ndarray): Community labels.
        num_points (int): Number of data points to visualize.

    Returns:
        fig - Plot figure of UMAP visualization embedding with community labels.
    """
    num = np.unique(community_label).shape[0]
    fig = plt.figure(1)
    plt.scatter(embed[:num_points,0], embed[:num_points,1],  c=community_label[:num_points], cmap='Spectral', s=2, alpha=.7)
    #plt.colorbar(boundaries=np.arange(num+1)-0.5).set_ticks(np.arange(num))
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    return fig


def visualization(
    config: Union[str, Path],
    label: Optional[str] = None
) -> None:
    """
    Visualize UMAP embeddings based on configuration settings.

    Args:
        config (Union[str, Path]): Path to the configuration file.
        label (str, optional): Type of labels to visualize. Default is None.

    Returns:
        None - Plot Visualization of UMAP embeddings.
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    param = cfg['parametrization']

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
        path_to_file=os.path.join(cfg['project_path'],"results",file,"",model_name,"",param+'-'+str(n_cluster))

        try:
            embed = np.load(os.path.join(path_to_file,"","community","","umap_embedding_"+file+".npy"))
            num_points = cfg['num_points']
            if num_points > embed.shape[0]:
                num_points = embed.shape[0]
        except:
            if not os.path.exists(os.path.join(path_to_file,"community")):
                os.mkdir(os.path.join(path_to_file,"community"))
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

        print("Visualizing %d data points.. " %num_points)
        if label is None:
            output_figure = umap_vis(file, embed, num_points)
            fig_path = os.path.join(path_to_file,"community","umap_vis_label_none_"+file+".png")
            output_figure.savefig(fig_path)
            return output_figure

        if label == 'motif':
            motif_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_' + param + '_label_'+file+'.npy'))
            output_figure = umap_label_vis(file, embed, motif_label, n_cluster, num_points)
            fig_path = os.path.join(path_to_file,"community","umap_vis_motif_"+file+".png")
            output_figure.savefig(fig_path)
            return output_figure

        if label == "community":
            community_label = np.load(os.path.join(path_to_file,"","community","","community_label_"+file+".npy"))
            output_figure = umap_vis_comm(file, embed, community_label, num_points)
            fig_path = os.path.join(path_to_file,"community","umap_vis_community_"+file+".png")
            output_figure.savefig(fig_path)
            return output_figure











