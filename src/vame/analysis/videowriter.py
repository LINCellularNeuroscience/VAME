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
from pathlib import Path
import numpy as np
import cv2 as cv
import tqdm
from typing import Union
from vame.util.auxiliary import read_config
from vame.logging.redirect_stream import StreamToLogger
import imageio


def get_cluster_vid(
    cfg: dict,
    path_to_file: str,
    file: str,
    n_cluster: int,
    videoType: str,
    flag: str,
    output_video_type: str = '.mp4',
    tqdm_logger_stream:  StreamToLogger | None = None
) -> None:
    """
    Generate cluster videos.

    Args:
        cfg (dict): Configuration parameters.
        path_to_file (str): Path to the file.
        file (str): Name of the file.
        n_cluster (int): Number of clusters.
        videoType (str): Type of input video.
        flag (str): Flag indicating the type of video (motif or community).

    Returns:
        None - Generate cluster videos and save them to fs on project folder.
    """

    if output_video_type not in ['.mp4', '.avi']:
        raise ValueError("Output video type must be either '.avi' or '.mp4'.")

    param = cfg['parametrization']
    if flag == "motif":
        print("Motif videos getting created for "+file+" ...")
        labels = np.load(os.path.join(path_to_file,str(n_cluster)+'_' + param + '_label_'+file+'.npy'))
    if flag == "community":
        print("Community videos getting created for "+file+" ...")
        labels = np.load(os.path.join(path_to_file,"community",'community_label_'+file+'.npy'))
    capture = cv.VideoCapture(os.path.join(cfg['project_path'],"videos",file+videoType))

    if capture.isOpened():
        width  = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
        fps = 25#capture.get(cv.CAP_PROP_FPS)

    cluster_start = cfg['time_window'] / 2

    for cluster in range(n_cluster):
        print('Cluster: %d' %(cluster))
        cluster_lbl = np.where(labels == cluster)
        cluster_lbl = cluster_lbl[0]
        if not cluster_lbl.size:
            print('Cluster is empty')
            continue

        if flag == "motif":
            output = os.path.join(path_to_file,"cluster_videos",file+f'-motif_%d{output_video_type}' %cluster)
        if flag == "community":
            output = os.path.join(path_to_file,"community_videos",file+f'-community_%d{output_video_type}' %cluster)

        if output_video_type == '.avi':
            codec = cv.VideoWriter_fourcc("M", "J", "P", "G")
            video_writer = cv.VideoWriter(output, codec, fps, (int(width), int(height)))
        elif output_video_type == '.mp4':
            video_writer = imageio.get_writer(output, fps=fps, codec='h264', macro_block_size=None)


        if len(cluster_lbl) < cfg['length_of_motif_video']:
            vid_length = len(cluster_lbl)
        else:
            vid_length = cfg['length_of_motif_video']

        for num in tqdm.tqdm(range(vid_length), file=tqdm_logger_stream):
            idx = cluster_lbl[num]
            capture.set(1,idx+cluster_start)
            ret, frame = capture.read()
            if output_video_type == '.avi':
                video_writer.write(frame)
            elif output_video_type == '.mp4':
                video_writer.append_data(frame)
        if output_video_type == '.avi':
            video_writer.release()
        elif output_video_type == '.mp4':
            video_writer.close()
    capture.release()


def motif_videos(
    config: Union[str, Path],
    videoType: str = '.mp4',
    output_video_type: str = '.mp4',
    save_logs: bool = False
) -> None:
    """
    Generate motif videos.

    Args:
        config (Union[str, Path]): Path to the configuration file.
        videoType (str, optional): Type of video. Default is '.mp4'.
        output_video_type (str, optional): Type of output video. Default is '.mp4'.

    Returns:
        None - Generate motif videos and save them to filesystem on project cluster_videos folder.
    """
    try:
        redirect_stream = StreamToLogger()
        tqdm_logger_stream = None
        config_file = Path(config).resolve()
        cfg = read_config(config_file)
        if save_logs:
            log_path = Path(cfg['project_path']) / 'logs' / 'motif_videos.log'
            redirect_stream.add_file_handler(log_path)
            tqdm_logger_stream = redirect_stream
        model_name = cfg['model_name']
        n_cluster = cfg['n_cluster']
        param = cfg['parametrization']
        flag = 'motif'

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

        print("Cluster size is: %d " %n_cluster)
        for file in files:
            path_to_file=os.path.join(cfg['project_path'],"results",file,model_name,param+'-'+str(n_cluster),"")
            if not os.path.exists(os.path.join(path_to_file,"cluster_videos")):
                os.mkdir(os.path.join(path_to_file,"cluster_videos"))

            get_cluster_vid(cfg, path_to_file, file, n_cluster, videoType, flag, output_video_type=output_video_type, tqdm_logger_stream=tqdm_logger_stream)

        print("All videos have been created!")
    except Exception as e:
        redirect_stream.logger.exception(f"Error in motif_videos: {e}")
        raise e
    finally:
        redirect_stream.stop()

def community_videos(config: Union[str, Path], videoType: str = '.mp4', save_logs: bool = False) -> None:
    """
    Generate community videos.

    Args:
        config (Union[str, Path]): Path to the configuration file.
        videoType (str, optional): Type of video. Default is '.mp4'.

    Returns:
        None - Generate community videos and save them to filesystem on project community_videos folder.
    """
    try:
        redirect_stream = StreamToLogger()
        tqdm_logger_stream = None
        config_file = Path(config).resolve()
        cfg = read_config(config_file)

        if save_logs:
            log_path = Path(cfg['project_path']) / 'logs' / 'community_videos.log'
            redirect_stream.add_file_handler(log_path)
            tqdm_logger_stream = redirect_stream
        model_name = cfg['model_name']
        n_cluster = cfg['n_cluster']
        param = cfg['parametrization']
        flag = 'community'

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

        print("Cluster size is: %d " %n_cluster)
        for file in files:
            path_to_file=os.path.join(cfg['project_path'],"results",file,model_name,param+'-'+str(n_cluster),"")
            if not os.path.exists(os.path.join(path_to_file,"community_videos")):
                os.mkdir(os.path.join(path_to_file,"community_videos"))

            get_cluster_vid(cfg, path_to_file, file, n_cluster, videoType, flag, tqdm_logger_stream=tqdm_logger_stream)

        print("All videos have been created!")

    except Exception as e:
        redirect_stream.logger.exception(f"Error in community_videos: {e}")
        raise e
    finally:
        redirect_stream.stop()