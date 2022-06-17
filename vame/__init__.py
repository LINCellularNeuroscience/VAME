#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""
import sys
sys.dont_write_bytecode = True

from vame.initialize_project import init_new_project
from vame.model import create_trainset
from vame.model import train_model
from vame.model import evaluate_model
from vame.analysis import pose_segmentation
from vame.analysis import motif_videos
from vame.analysis import community
from vame.analysis import community_videos
from vame.analysis import visualization
from vame.analysis import generative_model
from vame.analysis import gif
from vame.analysis import vis_rnn_network
from vame.util.csv_to_npy import csv_to_numpy
from vame.util.align_egocentrical import egocentric_alignment
from vame.util import auxiliary
from vame.util.auxiliary import update_config

