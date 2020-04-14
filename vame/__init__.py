#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:32:12 2019

@author: luxemk
"""
import sys
sys.dont_write_bytecode = True
      
from vame.initalize_project import init_new_project
from vame.model import create_trainset
from vame.model import rnn_model
from vame.model import evaluate_model