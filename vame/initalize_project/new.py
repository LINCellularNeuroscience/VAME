#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:40:24 2019

@author: luxemk
"""

import os
from pathlib import Path
import shutil

def init_new_project(project, videos, pose_files=False, working_directory=None, videotype='.mp4'):
    from datetime import datetime as dt
    from VAME.util import auxiliary
    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    year = date.year
    d = str(month[0:3]+str(day))
    date = dt.today().strftime('%Y-%m-%d')
    
    if working_directory == None:
        working_directory = '.'
        
    wd = Path(working_directory).resolve()
    project_name = '{pn}-{date}'.format(pn=project, date=d+'-'+str(year)) 
    
    project_path = wd / project_name
    
    
    if project_path.exists():
        print('Project "{}" already exists!'.format(project_path))
        return
    
    video_path = project_path / 'videos'
    data_path = project_path / 'data'
    results_path = project_path / 'results'
    model_path = project_path / 'model'
    
    for p in [video_path, data_path, results_path, model_path]:
        p.mkdir(parents=True)
        print('Created "{}"'.format(p))
    
    vids = []
    for i in videos:
        #Check if it is a folder
        if os.path.isdir(i):
            vids_in_dir = [os.path.join(i,vp) for vp in os.listdir(i) if videotype in vp]
            vids = vids + vids_in_dir
            if len(vids_in_dir)==0:
                print("No videos found in",i)
                print("Perhaps change the videotype, which is currently set to:", videotype)
            else:
                videos = vids
                print(len(vids_in_dir)," videos from the directory" ,i, "were added to the project.")
        else:
            if os.path.isfile(i):
                vids = vids + [i]
            videos = vids
            
            
    videos = [Path(vp) for vp in videos]
    video_names = []
    dirs_data = [data_path/Path(i.stem) for i in videos]
    for p in dirs_data:
        """
        Creates directory under data
        """
        p.mkdir(parents = True, exist_ok = True)
        video_names.append(p.stem)
        
    dirs_results = [results_path/Path(i.stem) for i in videos]
    for p in dirs_results:
        """
        Creates directory under results
        """
        p.mkdir(parents = True, exist_ok = True)
        
    destinations = [video_path.joinpath(vp.name) for vp in videos]
    
    if pose_files == True:
        os.mkdir(str(project_path)+'/'+'videos/pose_estimation/')
           
    print("Copying the videos")
    for src, dst in zip(videos, destinations):
        shutil.copy(os.fspath(src),os.fspath(dst))

    cfg_file,ruamelFile = auxiliary.create_config_template()
    cfg_file
    
    cfg_file['Project']=str(project)
    cfg_file['project_path']=str(project_path)+'/'
    cfg_file['TestFraction']=0.2
    cfg_file['video_sets']=video_names
    cfg_file['resnet']='50'
    cfg_file['resnet_pretrained']=True
    cfg_file['batch_size_spatial']=64
    cfg_file['Epochs_spatial']=500 
    cfg_file['ZDIMS_spatial']=25 
    cfg_file['BETA_spatial']=1
    cfg_file['anneal_function']='linear'
    cfg_file['Learning_rate_spatial']=1e-4
    cfg_file['batch_size_temporal']=128 
    cfg_file['Epochs_temporal']=500 
    cfg_file['rnn_model']='GRU'
    cfg_file['BETA_temporal']=1
    cfg_file['ZDIMS_temporal']=5
    cfg_file['Learning_rate_temporal']=1e-4
    cfg_file['temporal_window']=60
    cfg_file['future_decoder']=0
    cfg_file['future_steps']=10
    cfg_file['model_convergence']=20
    cfg_file['num_features']=25

    projconfigfile=os.path.join(str(project_path),'config.yaml')
    # Write dictionary to yaml  config file
    auxiliary.write_config(projconfigfile,cfg_file)
    
    print('A VAME project has been created. Please manually add your DLC file \n'
          'for the video to the folder videos. This will be automatic in later version. \n'
          '\n'
          'Next use vame.align() if you want to align and crop the animal frames. \n'
          'Otherwise, if you working with a fixed animal, you can skip this and \n'
          'use the vame.temporal() or vame.spatial() function.')
    
    return projconfigfile
    
    
    
    
    
    
    
    
    
            
            
            