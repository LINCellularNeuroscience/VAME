#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:53:16 2019

@author: luxemk
"""

import sys
sys.dont_write_bytecode = True

from vame.model.create_training import create_trainset
from vame.model.dataloader import SEQUENCE_DATASET
from vame.model.rnn_vae import rnn_model
#from VAME.model.evaluate import evaluate_model

