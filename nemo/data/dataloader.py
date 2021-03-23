import os
import logging

import cv2
import pandas as pd
import torch 
from torch.utils.data import Dataset

from nemo.data.utils import get_fpaths_in_dir


logging.basicConfig(
    format='%(levelname)s -- %(asctime)s -- %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p', 
    level = logging.INFO
)


class TrialAvgNeuralDataset(Dataset):
    def __init__(self, data_dir, stimuli, cre_lines = None, stim_height = 32, 
                 stim_width = 64):
        '''
        Dataset generator for trial avg trace dataset. 

        Args:
            data_dir (str): The path to the data directory. Should contain both a subdir 
            called NeuralData and one called Stimuli.
            stimuli (list): Stimuli to use in this experiment (i.e. natural_movie_one,
                natural_scenes, etc.).
            cre_lines (list): Keep cells if cre line is in cre_lines.
            stim_height (int): Desired height of the stimulus.
            stim_width (int): Desired width of the stimulus.
        '''

        self.data_dir = data_dir
        self.stimuli = stimuli
        self.cre_lines = cre_lines
        self.stim_height = stim_height
        self.stim_width = stim_width

        self.neural_data_dir = os.path.join(data_dir, 'NeuralData')
        self.stimuli_dir = os.path.join(data_dir, 'Stimuli')
        self.stim_frame_fpaths = dict(
            zip(
                [stim for stim in stimuli], 
                [get_fpaths_in_dir(os.path.join(self.stimuli_dir, stim)) for stim in stimuli]
            )
        )

        self.load_data()


    def load_data(self):
        # read the file and select out the desired stimuli
        data = pd.read_hdf(os.path.join(self.neural_data_dir, 'dff.h5'))
        data = data[data.stimulus.isin(self.stimuli)]

        # keep cells with desired cre lines if given 
        if self.cre_lines is not None:
            cre_line_df = pd.read_hdf(os.path.join(self.neural_data_dir, 'cre_line.h5'))
            keep_conts = [col for col in cre_line_df.columns if cre_line_df[col].to_list()[0] in self.cre_lines]
            keep_cols = [col for col in data.columns if col.split('_')[-1] in keep_conts]
        else:
            keep_cols = [col for col in data.columns if col.split('_')[0].isdigit()]

        # get trial avgs by stimulus and frame number
        data = data[keep_cols + ['frame', 'stimulus']]
        data = data.groupby(['stimulus', 'frame']).mean().reset_index()
        self.data = data.dropna(axis = 1)

        logging.info('DATA LOADER INITIALIZED')
        logging.info('   - NEURAL DATA DIR: {}'.format(self.neural_data_dir))
        logging.info('   - STIMULI DATA DIR: {}'.format(self.stimuli_dir))
        logging.info('   - STIMULI: {}'.format(self.stimuli))
        logging.info('   - CRE LINES: {}'.format(self.cre_lines))
        logging.info('   - NUM. CELLS: {}'.format(len(self.data.columns) - 2))
        logging.info('   - NUM. STIMULUS FRAMES: {}'.format(len(self.data)))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        stimulus = self.data.stimulus.to_list()[idx]
        frame_num = int(self.data.frame.to_list()[idx])
        
        dff = torch.Tensor(self.data.iloc[idx, 2:].to_list())
        
        frame_fpath = self.stim_frame_fpaths[stimulus][frame_num]
        frame = cv2.imread(frame_fpath, cv2.IMREAD_GRAYSCALE)
        frame = cv2.resize(frame, (self.stim_width, self.stim_height))
        
        return torch.from_numpy(frame), dff