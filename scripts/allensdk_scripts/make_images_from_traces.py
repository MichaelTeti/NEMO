'''
Script to take in the .txt files with each cell's traces created by the
extract_neurodata.py script and turn the traces into an image to make it easy
for petavision to use.
'''

import os, sys

from argparse import ArgumentParser
import cv2
import numpy as np

from NEM.utils.image_utils import save_vid_array_as_frames
from NEM.utils.general_utils import (
    get_fpaths_in_dir,
    read_csv,
    str2float_list,
    find_common_vals_in_lists
)



if __name__ == '__main__':
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
        help = 'Directory to save the trace images in.'
    )
    parser.add_argument(
        '--stimuli',
        type = str,
        nargs = '+',
        required = True,
        help = "Which stimuli's traces to use?"
    )
    parser.add_argument(
        '--common_cells',
        action = 'store_true',
        help = 'If specified, will only use cells common to all stimuli found.'
    )
    parser.add_argument(
        '--trial_averaged',
        action = 'store_true',
        help = 'If specified, trial-averaged responses will be used.'
    )

    args = parser.parse_args()

    assert(os.path.isdir(args.trace_dir)), \
        'trace_dir does not exist or is not a directory.'
    assert(os.listdir(args.trace_dir) != []), \
        'trace_dir is empty.'

    trace_fpaths = [fpath for stimulus in args.stimuli for fpath in get_fpaths_in_dir(args.trace_dir, key = stimulus)]
    stimuli = [os.path.splitext(os.path.split(fpath)[1])[0] for fpath in trace_fpaths]
    save_dirs = [os.path.join(args.save_dir, stimulus) for stimulus in stimuli]
    for save_dir in save_dirs: os.makedirs(save_dir, exist_ok = True)
    dataset = [read_csv(fpath) for fpath in trace_fpaths]
    img_fnames = [dset[0][1:] for dset in dataset]
    dataset = [dset[1:] for dset in dataset]
    dataset = [str2float_list(dset, start_ind = 1) for dset in dataset]

    # keep only cells that are in all stimuli / same experiment
    # first item in each row is container/experiment/cell id/cell ind
    if args.common_cells:
        meta = [[''.join(row[0].split('/')[1:]) for row in dset] for dset in dataset]
        common_cells = find_common_vals_in_lists(*meta)
        dataset = [[row for row in dset if ''.join(row[0].split('/')[1:]) in common_cells] for dset in dataset]

    # separate the meta data from the traces
    dataset = [[row[1:] for row in dset] for dset in dataset]

    # turn data in to np.ndarrays
    dataset = [np.array(dset, dtype = np.float32) for dset in dataset]

    # compute trial-averaged responses
    if args.trial_averaged:
        trial_nums = [[int(img_fname.split('.')[0].split('_')[1]) for img_fname in dset] for dset in img_fnames]
        n_trials = [max(dset) + 1 for dset in trial_nums]
        img_nums = [[int(img_fname.split('.')[0].split('_')[0]) for img_fname in dset] for dset in img_fnames]
        n_imgs = [max(dset) + 1 for dset in img_nums]
        avg_responses = [np.zeros([dset.shape[0], n_imgs_stim], dtype = np.float32) for dset, n_imgs_stim in zip(dataset, n_imgs)]
        for i_dset, (dset, n_trials_stim, n_imgs_stim, avg_responses_stim) in enumerate(zip(dataset, n_trials, n_imgs, avg_responses)):
            for trace_idx in range(0, n_imgs_stim * n_trials_stim, n_imgs_stim):
                avg_responses_stim += dset[:, trace_idx:trace_idx + n_imgs_stim]
            avg_responses_stim = np.uint8(avg_responses_stim / n_trials_stim * 255)
            dataset[i_dset] = avg_responses_stim.transpose()[..., None]
    else:
        raise NotImplementedError('non-trial-averaged not implemented yet.')


    save_vid_array_as_frames(list(zip(dataset, save_dirs)))
