import os
import logging
from multiprocessing import cpu_count

import cv2
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader

from nemo.data.utils import get_fpaths_in_dir


logging.basicConfig(
    format='%(levelname)s -- %(asctime)s -- %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p', 
    level = logging.INFO
)


class TrialAvgNeuralDataset(Dataset):
    def __init__(self, data_dir):
        '''
        Dataset generator for trial avg trace dataset. 

        Args:
            dff_fpath (str): The path to dff.h5 file to use.
            stimuli_dir (list): Path to the .
        '''

        self.data_dir = data_dir
        self.neural_data_dir = os.path.join(data_dir, 'NeuralData')
        self.stimuli_dir = os.path.join(data_dir, 'Stimuli')

        # self.img_fpaths = dict(
        #     zip(
        #         [stim for stim in stimuli], 
        #         [get_fpaths_in_dir(os.path.join(self.stimuli_dir, stim)) for stim in self.stim_dirs]
        #     )
        # )

        # logging.info('DATA LOADER INITIALIZED')
        # logging.info('   - NEURAL DATA DIR: {}'.format(self.neural_data_dir))
        # logging.info('   - STIMULI DATA DIR: {}'.format(self.stimuli_dir))
        # logging.info('   - STIMULI: {}'.format(self.stimuli))
        # logging.info('   - CRE LINES: {}'.format(self.cre_lines))
        # logging.info('   - NUM. CELLS: {}'.format(len(self.df.columns[self.dff_inds])))
        # logging.info('   - NUM. STIMULUS FRAMES: {}'.format(len(self.df)))


    def load_data(self):
        pass


    def __len__(self):
        pass


    def __getitem__(self, idx):
        pass