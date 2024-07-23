#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0

The following code is adapted from:

DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
from pathlib import Path
import shutil
from datetime import datetime as dt
from vame.util.auxiliary import write_config
from typing import List
from vame.schemas.project import ProjectSchema
from vame.schemas.states import VAMEPipelineStatesSchema
import json
from vame.logging.logger import VameLogger

logger_config = VameLogger(__name__)
logger = logger_config.logger



def init_new_project(
    project: str,
    videos: List[str],
    poses_estimations: List[str],
    working_directory: str = '.',
    videotype: str = '.mp4'
) -> str:
    """Creates a new VAME project with the given parameters.

    Args:
        project (str): Project name.
        videos (List[str]): List of videos paths to be used in the project. E.g. ['./sample_data/Session001.mp4']
        poses_estimations (List[str]): List of pose estimation files paths to be used in the project. E.g. ['./sample_data/pose estimation/Session001.csv']        working_directory (str, optional): _description_. Defaults to None.
        videotype (str, optional): Video extension (.mp4 or .avi). Defaults to '.mp4'.

    Returns:
        projconfigfile (str): Path to the new vame project config file.
    """

    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    year = date.year
    d = str(month[0:3]+str(day))
    date = dt.today().strftime('%Y-%m-%d')

    wd = Path(working_directory).resolve()
    project_name = '{pn}-{date}'.format(pn=project, date=d+'-'+str(year))

    project_path = wd / project_name
    if project_path.exists():
        logger.info('Project "{}" already exists!'.format(project_path))
        projconfigfile = os.path.join(str(project_path),'config.yaml')
        return projconfigfile

    video_path = project_path / 'videos'
    data_path = project_path / 'data'
    results_path = project_path / 'results'
    model_path = project_path / 'model'

    for p in [video_path, data_path, results_path, model_path]:
        p.mkdir(parents=True)
        logger.info('Created "{}"'.format(p))

    vids = []
    for i in videos:
        #Check if it is a folder
        if os.path.isdir(i):
            vids_in_dir = [os.path.join(i,vp) for vp in os.listdir(i) if videotype in vp]
            vids = vids + vids_in_dir
            if len(vids_in_dir)==0:
                logger.info(f"No videos found in {i}")
                logger.info(f"Perhaps change the videotype, which is currently set to: {videotype}")
            else:
                videos = vids
                logger.info(
                    f"{len(vids_in_dir)} videos from the directory {i} were added to the project."
                )
        else:
            if os.path.isfile(i):
                vids = vids + [i]
            videos = vids

    pose_estimations_paths = []
    for pose_estimation_path in poses_estimations:
        if os.path.isdir(pose_estimation_path):
            pose_estimation_files = [os.path.join(pose_estimation_path, p) for p in os.listdir(pose_estimation_path) if '.csv' in p]
            pose_estimations_paths.extend(pose_estimation_files)
        else:
            pose_estimations_paths.append(pose_estimation_path)

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

    os.mkdir(str(project_path)+'/'+'videos/pose_estimation/')
    os.mkdir(str(project_path)+'/model/pretrained_model')

    logger.info("Copying the videos \n")
    for src, dst in zip(videos, destinations):
        shutil.copy(os.fspath(src),os.fspath(dst))

    logger.info("Copying pose estimation files\n")
    for src, dst in zip(pose_estimations_paths, [str(project_path)+'/videos/pose_estimation/'+Path(p).name for p in pose_estimations_paths]):
        logger.info(f'Copying {src} to {dst}')
        shutil.copy(os.fspath(src),os.fspath(dst))

    new_project = ProjectSchema(
        Project=str(project),
        project_path=str(project_path),
        video_sets=video_names
    )
    cfg_data = new_project.model_dump()

    projconfigfile=os.path.join(str(project_path), 'config.yaml')
    # Write dictionary to yaml  config file
    write_config(projconfigfile, cfg_data)

    vame_pipeline_default_schema = VAMEPipelineStatesSchema()
    vame_pipeline_default_schema_path = Path(project_path) / 'states/states.json'
    if not vame_pipeline_default_schema_path.parent.exists():
        vame_pipeline_default_schema_path.parent.mkdir(parents=True)
    with open(vame_pipeline_default_schema_path, 'w') as f:
        json.dump(vame_pipeline_default_schema.model_dump(), f, indent=4)

    logger.info('A VAME project has been created. \n')
    logger.info('Now its time to prepare your data for VAME. '
          'The first step is to move your pose .csv file (e.g. DeepLabCut .csv) into the '
          '//YOUR//VAME//PROJECT//videos//pose_estimation folder. From here you can call '
          'either the function vame.egocentric_alignment() or if your data is by design egocentric '
          'call vame.csv_to_numpy(). This will prepare the data in .csv into the right format to start '
          'working with VAME.')

    return projconfigfile
