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

from vame.analysis.pose_segmentation import pose_segmentation
from vame.analysis.videowriter import motif_videos, community_videos
from vame.analysis.community_analysis import community
from vame.analysis.umap import visualization
from vame.analysis.generative_functions import generative_model
from vame.analysis.gif_creator import gif

