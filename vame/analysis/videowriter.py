#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:36:49 2019

@author: luxemk
"""

import os
from pathlib import Path
import numpy as np
import cv2 as cv

from vame.util.auxiliary import read_config


def get_cluster_vid(cfg, path_to_file, file, n_cluster):
    print("Videos get created for "+file+" ...")
    labels = np.load(path_to_file+'/'+str(n_cluster)+'_km_label_'+file+'.npy')
    capture = cv.VideoCapture(cfg['project_path']+'videos/'+file+'.mp4')
    
    if capture.isOpened(): 
        width  = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)  
#        print('width, height:', width, height)
    
        fps = capture.get(cv.CAP_PROP_FPS)
#        print('fps:', fps)  
    
    cluster_start = cfg['time_window'] / 2
    for cluster in range(n_cluster):
        print('Cluster: %d' %(cluster))
        cluster_lbl = np.where(labels == cluster)
        cluster_lbl = cluster_lbl[0]
    
        output = path_to_file+'/cluster_videos/'+file+'motif_%d.avi' %cluster
        video = cv.VideoWriter(output, cv.VideoWriter_fourcc('M','J','P','G'), fps, (int(width), int(height)))
        
        if len(cluster_lbl) < cfg['lenght_of_motif_video']:
            vid_length = len(cluster_lbl)
        else:
            vid_length = cfg['lenght_of_motif_video']
        
        for num in range(vid_length):
            idx = cluster_lbl[num]
            capture.set(1,idx+cluster_start)
            ret, frame = capture.read()
            video.write(frame)
                
        video.release()     
    capture.release()
    
    
def motif_videos(config, model_name, cluster_method="kmeans", n_cluster=[30]):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
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
    
    for cluster in n_cluster:
        print("Cluster size %d " %cluster)
        for file in files:
            path_to_file=cfg['project_path']+'results/'+file+'/'+model_name+'/'+cluster_method+'-'+str(cluster)
            
            if not os.path.exists(path_to_file+'/cluster_videos/'):
                    os.mkdir(path_to_file+'/cluster_videos/')
            
            get_cluster_vid(cfg, path_to_file, file, cluster)
        





