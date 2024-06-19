#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from vame.logging.logger import VameLogger


class SEQUENCE_DATASET(Dataset):
    def __init__(self, path_to_file: str, data: str, train: bool, temporal_window: int, **kwargs) -> None:
        """Initialize the Sequence Dataset.

        Args:
            path_to_file (str): Path to the dataset files.
            data (str): Name of the data file.
            train (bool): Flag indicating whether it's training data.
            temporal_window (int): Size of the temporal window.

        Returns:
            None
        """
        self.logger_config = kwargs.get('logger_config', VameLogger(__name__))
        self.logger = self.logger_config.logger

        self.temporal_window = temporal_window
        self.X = np.load(path_to_file+data)
        if self.X.shape[0] > self.X.shape[1]:
            self.X=self.X.T

        self.data_points = len(self.X[0,:])

        if train and not os.path.exists(os.path.join(path_to_file,'seq_mean.npy')):
            self.logger.info("Compute mean and std for temporal dataset.")
            self.mean = np.mean(self.X)
            self.std = np.std(self.X)
            np.save(path_to_file+'seq_mean.npy', self.mean)
            np.save(path_to_file+'seq_std.npy', self.std)
        else:
            self.mean = np.load(path_to_file+'seq_mean.npy')
            self.std = np.load(path_to_file+'seq_std.npy')

        if train:
            self.logger.info('Initialize train data. Datapoints %d' %self.data_points)
        else:
            self.logger.info('Initialize test data. Datapoints %d' %self.data_points)

    def __len__(self) -> int:
        """Return the number of data points.

        Returns:
            int: Number of data points.
        """
        return self.data_points

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get a normalized sequence at the specified index.

        Args:
            index (int): Index of the item.

        Returns:
            torch.Tensor: Normalized sequence data at the specified index.
        """
        temp_window = self.temporal_window

        nf = self.data_points
        start = np.random.choice(nf-temp_window)
        end = start+temp_window

        sequence = self.X[:,start:end]

        sequence = (sequence-self.mean)/self.std

        return torch.from_numpy(sequence)





