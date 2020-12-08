from argparse import ArgumentParser
import json
import os, sys

import h5py
from imageio import imread
import numpy as np
import pandas as pd
import torch
#from torchvision import transforms, utils
from torch.utils.data import Dataset

from NEMO.utils.general_utils import read_csv


def load_trial_averaged_traces(fpath, n_frames_in_time = 9):
    traces = pd.read_csv(fpath)
    traces = traces.iloc[:, n_frames_in_time - 1:].to_numpy()[0]
    return traces


def save_args(args, save_dir):
    arg_dict = vars(args)
    with open(os.path.join(save_dir, 'args.txt'), 'w') as fp:
        json.dump(arg_dict, fp, sort_keys = True, indent = 4)


def create_temporal_design_mat(vid_frame_array, n_frames_in_time = 9):
    n_frames, h, w = vid_frame_array.shape
    cat_list = [vid_frame_array[i:n_frames - (n_frames_in_time - i - 1)][..., None] for i in range(n_frames_in_time)] 
    design_mat = np.concatenate(cat_list, axis = 3)
    return design_mat
        


def shuffle_data(self, preds, responses):
    '''
    Shuffles the rows of the design matrix and the responses. 

    Args:
        preds (np.ndarray): The N x M dimensional design matrix with N samples and M predictors.
        responses (np.ndarray): The N-dimensional response vector corresponding to preds. 

    Returns:
        preds_shuffled (np.ndarray): The N x M dimensional design matrix with permuted rows. 
        responses_shuffled (np.ndarray): The N-dimensional response vector with permuted values matching 
            preds_shuffled. 
    '''

    data = np.concatenate((preds, responses[:, None]), 1)
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1]


class NeuralResponseDataset(Dataset):
    '''
    TODO write stuff here
    '''

    def __init__(self, img_dir, fpath_traces, fpath_pupil_coords, dset_type, seed = 12345):
        self.img_dir = img_dir
        np.random.seed(seed)

        # read the .csv files with the traces and pupil coordinates
        traces = read_csv(fpath_traces)
        pupil_coords = read_csv(fpath_pupil_coords)

        # get the frame_nums and remove this from the dataset
        self.frame_nums = traces[0][1:]
        traces = traces[1:]
        pupil_coords = pupil_coords[1:]

        # get the number of frames in the video for splitting up into train / val sets
        self.n_frames_in_video = max([int(frame_num.split('.')[0].split('_')[0]) for frame_num in self.frame_nums]) + 1

        # get the number of trials
        self.n_trials = max([int(frame_num.split('.')[0].split('_')[1]) for frame_num in self.frame_nums]) + 1

        # get cell / experiment data and then remove that column from the data
        self.meta_data = [row[0] for row in traces]
        traces = [row[1:] for row in traces]
        pupil_coords = [row[1:] for row in pupil_coords]

        # turn values into floats from strings
        traces = [[float(val) if val != 'nan' else np.nan for val in row] for row in traces]
        # need to separate the x and y values in pupil coords first
        pupil_coords = [[val.split('/') for val in row] for row in pupil_coords]
        pupil_coords = [[[float(coord) if coord != 'nan' else np.nan for coord in val] for val in row] for row in pupil_coords]

        # now turn into numpy arrays and transpose so the rows are samples and the columns are the cells
        self.traces = np.array(traces).transpose()
        self.pupil_coords = np.array(pupil_coords).transpose(1, 0, 2)

        # separate the training and validation samples such that each frame is only in one set even over different trials
        if dset_type == 'train':
            # index 80% of the frames
            inds = np.random.choice(self.n_frames_in_video, int(self.n_frames_in_video * 0.8), replace = False)
            # translate this to all other trials for the same video
            inds = np.concatenate([inds + self.n_frames_in_video * i_trial for i_trial in range(self.n_trials)])
            print(inds.shape, np.sort(inds)[:10])
            # index the data using these indices
            self.traces = self.traces[inds]
            self.pupil_coords = self.pupil_coords[inds]
            self.frame_nums = [frame_num for i_frame_num, frame_num in enumerate(self.frame_nums) if i_frame_num in inds]
        elif dset_type == 'val':
            # index 80% of the frames like before...should be same as train since we set the seed
            inds = np.random.choice(self.n_frames_in_video, int(self.n_frames_in_video * 0.8), replace = False)
            # take indices that are NOT in that set
            inds = np.array([i for i in range(self.n_frames_in_video) if i not in inds])
            # translate this to all of the repeats of the video
            inds = np.concatenate([inds + self.n_frames_in_video * i_trial for i_trial in range(self.n_trials)])
            # index the data using these indices
            self.traces = self.traces[inds]
            self.pupil_coords = self.pupil_coords[inds]
            self.frame_nums = [frame_num for i_frame_num, frame_num in enumerate(self.frame_nums) if i_frame_num in inds]


    def __len__(self):
        return self.traces.shape[0]


    def __getitem__(self, idx):
        img_name, img_ext = os.path.splitext(self.frame_nums[idx])
        img_fname = img_name.split('_')[0] + img_ext
        img_fpath = os.path.join(self.img_dir, img_fname)
        img = imread(img_fpath)
        traces = self.traces[idx]
        data = {'image': img, 'traces': traces}
        data = ToTensor()(data)

        return data


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        img, traces = data['image'], data['traces']

        # add a channel to the image
        img = img[None, ...]

        return {'image': torch.from_numpy(img),
                'traces': torch.from_numpy(traces)}


def get_model_args():
    parser = ArgumentParser()
    parser.add_argument('--img_dir',
        type = str,
        required = True,
        help = 'Directory where the training images corresponding to the training responses are located.')
    parser.add_argument('--response_path',
        type = str,
        required = True,
        help = 'The path to the .txt file with the responses corresponding to the images in img_dir.')
    parser.add_argument('--pupil_coords_path',
        type = str,
        required = True,
        help = 'The path to the .txt file with the pupil coordinates corresponding to the images in img_dir.')

    parser.add_argument('--batch_size',
        type = int,
        default = 64,
        help = 'The batch size used for training. Default is 64.')
    parser.add_argument('--n_workers',
        type = int,
        default = 1,
        help = 'The number of workers for the dataloader. Default is 1.')

    return parser.parse_args()
