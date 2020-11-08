'''
Script to take in the .txt files with each cell's traces created by the
extract_neurodata.py script and turn the traces into an image to make it easy
for petavision to use.
'''

from argparse import ArgumentParser
import os, sys

import cv2
import numpy as np
import pandas as pd

from NEMO.utils.image_utils import save_vid_array_as_frames
from NEMO.utils.general_utils import (
    get_fpaths_in_dir,
    read_csv,
    str2float_list,
    find_common_vals_in_lists,
    get_intersection_col_vals
)


parser = ArgumentParser()
parser.add_argument(
    'trace_dir',
    type = str,
    help = 'Directory containing one or more .txt files with the normalized \
        traces for a specific stimulus.'
)
parser.add_argument(
    'save_dir',
    type = str,
    help = 'Directory to save the trial-averaged response files in.'
)
parser.add_argument(
    '--stimuli',
    type = str,
    nargs = '+',
    required = True,
    help = "Which stimuli's traces to use?"
)
parser.add_argument(
    '--session_types',
    choices = [
        'three_session_A',
        'three_session_B',
        'three_session_C',
        'three_session_C2'
    ],
    required = True,
    nargs = '+',
    help = 'Which sessions to extract data from.'
)
parser.add_argument(
    '--common_cells',
    action = 'store_true',
    help = 'If specified, will save trial-averaged responses for cells common to \
        all stimuli given.'
)

args = parser.parse_args()

assert(os.path.isdir(args.trace_dir)), \
    'trace_dir does not exist or is not a directory.'
assert(os.listdir(args.trace_dir) != []), \
    'trace_dir is empty.'

# get the list of all the fpaths for the trace files for all stimuli given
trace_fpaths = [fpath for stimulus in args.stimuli for fpath in get_fpaths_in_dir(args.trace_dir, key = stimulus)]

# read in each trace fpath
datasets = [pd.read_csv(fpath) for fpath in trace_fpaths]

# filter out session_types
datasets = [dataset[dataset['session_type'].isin(args.session_types)] for dataset in datasets]

# get cell_ids for cells common to all stimuli given and filter out the data given these cell_ids
if args.common_cells:
    common_cell_ids = get_intersection_col_vals(datasets, col_name = 'cell_id')
    datasets = [dataset[dataset['cell_id'].isin(common_cell_ids)] for dataset in datasets]

# let's go wide to long so we can more easily get the average per frame
datasets = [
    pd.melt(
        dataset,
        id_vars = dataset.columns[:7],
        var_name = 'img_fname',
        value_name = 'response'
    )
    for dataset in datasets
]

# compute trial averages and save
for i_dataset, dataset in enumerate(datasets):
    dataset_trial_avg = dataset.groupby(['cell_id', 'session_type', 'stimulus', 'img_fname'])['response'].mean()
    dataset_trial_avg = dataset_trial_avg.reset_index(name = 'Mean Response')

    dataset_trial_avg.to_csv(
        os.path.join(args.save_dir, '{}_trial_averaged.txt'.format(args.stimuli[i_dataset])),
        header = True,
        mode = 'w',
        index = False
    )
