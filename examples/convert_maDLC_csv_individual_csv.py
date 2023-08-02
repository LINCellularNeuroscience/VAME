# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:20:44 2022

@author: Charitha Omprakash, LIN Magdeburg, charitha.omprakash@lin-magdeburg.de 

This file converts a multi-animal DLC CSV to several single animal DLC files.
Those can be used as input to run VAME.
"""

import pandas, numpy as pd, np
import os
import glob
from pathlib import Path

def convert_multi_csv_to_individual_csv(csv_files_path):
    csvs = sorted(glob.glob(os.path.join(csv_files_path, '*.csv*')))
    
    for csv in csvs:
        fname = pd.read_csv(csv, header=[0,1,2], index_col=0, skiprows=1)
        individuals = fname.columns.get_level_values('individuals').unique()
        for ind in individuals:
            fname_temp = fname[ind]
            fname_temp_path = os.path.splitext(csv)[0] + '_' + ind + '.csv'
            fname_temp.to_csv(fname_temp_path, index=True, header=True)
