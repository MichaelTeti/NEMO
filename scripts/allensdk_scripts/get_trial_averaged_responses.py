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

args = parser.parse_args()

assert(os.path.isdir(args.trace_dir)), \
    'trace_dir does not exist or is not a directory.'
assert(os.listdir(args.trace_dir) != []), \
    'trace_dir is empty.'

# get the list of all the fpaths for the trace files for all stimuli and session_types given
for root, _, files in os.walk(args.trace_dir):
    if any([stimulus in root for stimulus in args.stimuli]):
        if any([session_type in root for session_type in args.session_types]):
            for file in files:
                # read in the data
                df = pd.read_csv(os.path.join(root, file))

                # get stimuli and session type
                stimulus = df['stimulus'].tolist()[0]
                session_type = df['session_type'].tolist()[0]
                cell_id = df['cell_id'].tolist()[0]

                # let's go wide to long so we can more easily get the average per frame
                df = pd.melt(
                    df,
                    id_vars = df.columns[:7],
                    var_name = 'img_fname',
                    value_name = 'response'
                )

                # compute averages over the 10 trials
                df_avg = df.groupby('img_fname')['response'].mean().reset_index(name = 'mean_response')
                df_avg['mean_response'] = df_avg['mean_response'].round(4)

                # put back to wide format so format lines up with non-trial-averaged traces
                df_avg = df_avg.transpose()
                df_avg.columns = df_avg.iloc[0, :].tolist()
                df_avg = df_avg.drop('img_fname').reset_index(drop = True)

                # save trial-averaged responses
                save_dir = os.path.join(args.save_dir, stimulus, session_type)
                os.makedirs(save_dir, exist_ok = True)
                df_avg.to_csv(
                    os.path.join(save_dir, 'cellID_{}.txt'.format(cell_id)),
                    index = False
                )

# python3 get_trial_averaged_responses.py \
#     ../../../BrainObservatoryData/ExtractedData/Traces/ \
#     ../../../BrainObservatoryData/ExtractedData/TrialAveragedTraces/ \
#     --stimuli natural_movie_one natural_movie_three \
#     --session_types three_session_A
