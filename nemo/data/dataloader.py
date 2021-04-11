import os
import logging

import cv2
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset

from nemo.data.preprocess.image import max_min_scale
from nemo.data.preprocess.trace import compute_trial_avgs
from nemo.data.utils import get_fpaths_in_dir
from nemo.model.analysis.metrics import signal_power, cc_max


logging.basicConfig(
    format='%(levelname)s -- %(asctime)s -- %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p', 
    level = logging.INFO
)


class TrialAvgNeuralDataset(Dataset):
    def __init__(self, data_dir, stimuli, cre_lines = None, cell_ids = None, 
                 stim_height = 32, stim_width = 64, n_frames = 1, col_transform = None,
                 img_transform = None):
        '''
        Dataset generator for trial avgeraged recordings. 

        Args:
            data_dir (str): The path to the data directory. Should contain both a subdir 
            called NeuralData and one called Stimuli.
            stimuli (list): Stimuli to use in this experiment (i.e. natural_movie_one,
                natural_scenes, etc.).
            cre_lines (list): Keep cells if cre line is in cre_lines.
            cell_ids (list): Keep cells if cell ID in cell_ids. 
            stim_height (int): Desired height of the stimulus.
            stim_width (int): Desired width of the stimulus.
            n_frames (int): Number of stimulus frames to load at a time.
            device (int): Device to put data on. Default is None.
            col_transform (func): Describes how to transform each column after taking trial
                averages.
            img_transform (func): A function that determines the processing that is 
                performed on each individual image after it is read in.
        '''

        self.data_dir = data_dir
        self.stimuli = stimuli
        self.cre_lines = cre_lines
        self.cell_ids = cell_ids
        self.stim_height = stim_height
        self.stim_width = stim_width
        self.n_frames = n_frames
        self.col_transform = col_transform
        self.img_transform = img_transform

        self.neural_data_dir = os.path.join(data_dir, 'NeuralData')
        self.stimuli_dir = os.path.join(data_dir, 'Stimuli')
        self.stim_frame_fpaths = dict(
            zip(
                [stim for stim in stimuli], 
                [get_fpaths_in_dir(os.path.join(self.stimuli_dir, stim)) for stim in stimuli]
            )
        )

        self.load_neural_data()

        if self.n_frames < 1:
            raise ValueError('n_frames should be > 1.')
        if self.n_frames > len(self.data):
            raise ValueError('n_frames should be < len(self.data) ({})'.format(len(self.data)))


    def load_neural_data(self):
        ''' Load recordings and process them '''

        # read the file and select out the desired stimuli
        data = pd.read_hdf(os.path.join(self.neural_data_dir, 'dff.h5'))
        data = data[data.stimulus.isin(self.stimuli)]

        # get rid of frame -1 (blank gray frame) for this
        data = data[data.frame != -1]

        # get columns that correspond to cell responses, not stimulus info
        keep_cols = [col for col in data.columns if col.split('_')[0].isdigit()]
            
        # pull out cell columns if their container is in cre_lines if cre_lines was specified
        if self.cre_lines is not None:
            cre_line_df = pd.read_hdf(os.path.join(self.neural_data_dir, 'cre_line.h5'))
            keep_conts = [col for col in cre_line_df.columns if cre_line_df[col].to_list()[0] in self.cre_lines]
            keep_cols_cre = [col for col in data.columns if col.split('_')[-1] in keep_conts]
            keep_cols = list(set(keep_cols) & set(keep_cols_cre))

        # pull out cell columns if their cell ID is in cell_ids if cell_ids was specified
        if self.cell_ids is not None:
            keep_cols_id = [col for col in data.columns if col.split('_')[0] in self.cell_ids]
            keep_cols = list(set(keep_cols) & set(keep_cols_id))

        keep_cols = sorted(keep_cols, key = lambda col: col.split('_')[0])
        self.signal_power = signal_power(data[keep_cols + ['stimulus', 'frame', 'repeat', 'session_type']])
        data = data[['stimulus', 'frame'] + keep_cols]

        # get trial avgs by stimulus and frame number
        data = compute_trial_avgs(data)
        data = data.dropna(axis = 1)
        
        # get cc_max by cell-stimulus combo
        self.cc_max = cc_max(data.drop(columns = 'frame'), self.signal_power)

        # apply transform if provided
        if self.col_transform is not None:
            data.iloc[:, 2:] = self.col_transform(data.iloc[:, 2:])
            
        self.data = data
        self.cell_ids = [col.split('_')[0] for col in self.data.columns[2:]]
        self.cont_ids = [col.split('_')[1] for col in self.data.columns[2:]]

        logging.info('DATASET INITIALIZED')
        logging.info('   - NEURAL DATA DIR: {}'.format(self.neural_data_dir))
        logging.info('   - STIMULI DATA DIR: {}'.format(self.stimuli_dir))
        logging.info('   - STIMULI: {}'.format(self.stimuli))
        logging.info('   - CRE LINES: {}'.format(self.cre_lines))
        logging.info('   - NUM. CELLS: {}'.format(len(self.data.columns) - 2))
        logging.info('   - NUM. ANIMALS: {}'.format(len(list(set(self.cont_ids)))))
        logging.info('   - NUM. STIMULUS FRAMES: {}'.format(len(self.data)))


    def read_image(self, fpath):
        ''' Read image and resize to desired shape '''

        frame = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        frame = cv2.resize(frame, (self.stim_width, self.stim_height))

        if self.img_transform is not None:
            frame = self.img_transform(frame)

        return np.float32(max_min_scale(frame))[None, ...]


    def __len__(self):
        return len(self.data)


    def get_stimulus_and_frame(self, idx):
        ''' Get stimulus and frame given sample index '''

        stimuli = self.data.stimulus.to_list()[idx:idx + self.n_frames]
        frame_nums = [int(val) for val in self.data.frame.to_list()[idx:idx + self.n_frames]]

        return stimuli, frame_nums


    def __getitem__(self, idx):
        stimuli, frame_nums = self.get_stimulus_and_frame(idx)
        dff = [
            self.data.iloc[row, 2:].to_numpy(dtype = np.float32) \
            for row in range(idx, idx + self.n_frames)
        ]
        frame_fpaths = [
            self.stim_frame_fpaths[stim][frame_num] \
            for stim, frame_num in zip(stimuli, frame_nums)
        ]
        frames = [self.read_image(fpath) for fpath in frame_fpaths]
        
        return frames, dff