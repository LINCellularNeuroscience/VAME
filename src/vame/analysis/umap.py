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
from typing import Optional, Union, Dict
from vame.util.auxiliary import read_config
from vame.schemas.states import VisualizationFunctionSchema, save_state
from vame.logging.logger import VameLogger
from vame.schemas.project import Parametrizations


logger_config = VameLogger(__name__)
logger = logger_config.logger


def umap_embedding(cfg: dict, file: str, model_name: str, n_cluster: int, parametrization: str) -> np.ndarray:
    """Perform UMAP embedding for given file and parameters.

    Args:
        cfg (dict): Configuration parameters.
        file (str): File path.
        model_name (str): Model name.
        n_cluster (int): Number of clusters.
        parametrization (str): parametrization.

    Returns:
        np.ndarray: UMAP embedding.
    """
    reducer = umap.UMAP(
        n_components=2,
        min_dist=cfg['min_dist'],
        n_neighbors=cfg['n_neighbors'],
        random_state=cfg['random_state']
    )

    logger.info("UMAP calculation for file %s" %file)

    folder = os.path.join(cfg['project_path'],"results",file,model_name, parametrization +'-'+str(n_cluster),"")
    latent_vector = np.load(os.path.join(folder,'latent_vector_'+file+'.npy'))

    num_points = cfg['num_points']
    if num_points > latent_vector.shape[0]:
        num_points = latent_vector.shape[0]
    logger.info("Embedding %d data points.." %num_points)

    embed = reducer.fit_transform(latent_vector[:num_points,:])
    np.save(os.path.join(folder,"community","umap_embedding_"+file+'.npy'), embed)

    return embed


def umap_vis_community_labels(cfg: dict, embed: np.ndarray, community_labels_all: np.ndarray, save_path: str | None) -> None:
    """Create plotly visualizaton of UMAP embedding with community labels.

    Args:
        cfg (dict): Configuration parameters.
        embed (np.ndarray): UMAP embedding.
        community_labels_all (np.ndarray): Community labels.
        save_path: Path to save the plot. If None it will not save the plot.

    Returns:
        None
    """
    num_points = cfg['num_points']
    community_labels_all = np.asarray(community_labels_all)
    if num_points > community_labels_all.shape[0]:
        num_points = community_labels_all.shape[0]
    logger.info("Embedding %d data points.." %num_points)

    num = np.unique(community_labels_all)

    fig = plt.figure(1)
    plt.scatter(
        embed[:,0],
        embed[:,1],
        c=community_labels_all[:num_points],
        cmap='Spectral',
        s=2,
        alpha=1
    )
    plt.colorbar(boundaries=np.arange(np.max(num)+2)-0.5).set_ticks(np.arange(np.max(num)+1))
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)

    if save_path is not None:
        plt.savefig(save_path)
        return fig
    plt.show()
    return fig


def umap_vis(embed: np.ndarray, num_points: int) -> None:
    """
    Visualize UMAP embedding without labels.

    Args:
        embed (np.ndarray): UMAP embedding.
        num_points (int): Number of data points to visualize.

    Returns:
        None - Plot Visualization of UMAP embedding.
    """
    #plt.cla()
    #plt.clf()
    plt.close('all')
    fig = plt.figure(1)
    plt.scatter(embed[:num_points,0], embed[:num_points,1], s=2, alpha=.5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    return fig


def umap_label_vis(embed: np.ndarray, label: np.ndarray, n_cluster: int, num_points: int) -> None:
    """
    Visualize UMAP embedding with motif labels.

    Args:
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


def umap_vis_comm(embed: np.ndarray, community_label: np.ndarray, num_points: int) -> None:
    """
    Visualize UMAP embedding with community labels.

    Args:
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

@save_state(model=VisualizationFunctionSchema)
def visualization(
    config: Union[str, Path],
    parametrization: Parametrizations,
    label: Optional[str] = None,
    save_logs: bool = False
) -> None:
    """
    Visualize UMAP embeddings based on configuration settings.

    Args:
        config (Union[str, Path]): Path to the configuration file.
        label (str, optional): Type of labels to visualize. Default is None.

    Returns:
        None - Plot Visualization of UMAP embeddings.
    """
    try:
        config_file = Path(config).resolve()
        cfg = read_config(config_file)
        parametrizations = cfg['parametrizations']

        if parametrization not in parametrizations:
            raise ValueError(f"Parametrization {parametrization} not found in configuration file.")

        if save_logs:
            logs_path = Path(cfg['project_path']) / "logs" / 'visualization.log'
            logger_config.add_file_handler(logs_path)

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
            path_to_file=os.path.join(cfg['project_path'],"results",file,"",model_name,"",parametrization+'-'+str(n_cluster))

            try:
                embed = np.load(os.path.join(path_to_file,"","community","","umap_embedding_"+file+".npy"))
                num_points = cfg['num_points']
                if num_points > embed.shape[0]:
                    num_points = embed.shape[0]
            except Exception:
                if not os.path.exists(os.path.join(path_to_file,"community")):
                    os.mkdir(os.path.join(path_to_file,"community"))
                logger.info("Compute embedding for file %s" %file)
                embed = umap_embedding(cfg, file, model_name, n_cluster, parametrization)
                num_points = cfg['num_points']
                if num_points > embed.shape[0]:
                    num_points = embed.shape[0]

            if label is None:
                output_figure = umap_vis(embed, num_points)
                fig_path = os.path.join(path_to_file,"community","umap_vis_label_none_"+file+".png")
                output_figure.savefig(fig_path)

            if label == 'motif':
                motif_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_' + parametrization + '_label_'+file+'.npy'))
                output_figure = umap_label_vis(embed, motif_label, n_cluster, num_points)
                fig_path = os.path.join(path_to_file,"community","umap_vis_motif_"+file+".png")
                output_figure.savefig(fig_path)

            if label == "community":
                community_label = np.load(os.path.join(path_to_file,"","community","","community_label_"+file+".npy"))
                output_figure = umap_vis_comm(embed, community_label, num_points)
                fig_path = os.path.join(path_to_file,"community","umap_vis_community_"+file+".png")
                output_figure.savefig(fig_path)
    except Exception as e:
        logger.exception(str(e))
        raise e
    finally:
        logger_config.remove_file_handler()











