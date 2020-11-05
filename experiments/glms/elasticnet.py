from v1_response_prediction.utils.genutils import read_csv, write_h5, read_h5_as_array
from v1_response_prediction.utils.imutils import read_frames, max_min_scale

import os
import numpy as np
from argparse import ArgumentParser
from glob import glob
from imageio import imread, imwrite
import torch
import h5py
from sklearn.linear_model._glm import GeneralizedLinearRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV


parser = ArgumentParser()
parser.add_argument('frame_dir',
    type = str,
    help = 'Directory containing the images.')
parser.add_argument('response_fpath',
    type = str,
    help = 'Path to the .h5 file with the responses.')
parser.add_argument('pupil_loc_fpath',
    type = str,
    help = 'Path to the .h5 file with the pupil locations.')
parser.add_argument('save_path',
    type = str,
    help = 'Path to save the learned STRFs as an .h5 file.')
parser.add_argument('--response_key',
    type = str,
    help = 'Will only use responses and pupil locations if response_key is in the filename.')
parser.add_argument('--patch_size_x',
    type = int,
    default = 16,
    help = 'Width of the STRFs. Default is 16.')
parser.add_argument('--patch_size_y',
    type = int,
    default = 16,
    help = 'Height of the STRFs. Default is 16.')
parser.add_argument('--n_frames_in_time',
    type = int,
    default = 9,
    help = 'Number of sequential frames to model. Default is 9.')
parser.add_argument('--n_frames',
    type = int,
    help = 'Number of frames to read in and use for STRF estimation if not all of them.')
parser.add_argument('--feature_norm',
    action = 'store_true',
    help = 'Whether to perform feature normalization on the inputs.')
parser.add_argument('--max_iters',
    type = int,
    default = 1000,
    help = 'Maximum number of iterations for the model to do before quitting. Default is 1000.')
parser.add_argument('--downsample_training_set',
    type = int,
    help = 'Downsample the training samples by a factor of downsample_val (i.e. if this is 2, every other sample will be used).')
parser.add_argument('--n_folds',
    type = int,
    default = 5,
    help = 'The number of folds in cross validation. Default is 5.')
parser.add_argument('--n_jobs',
    type = int,
    help = 'The number of jobs to run during grid search. Default is 1.')
args = parser.parse_args()

psx, psy = args.patch_size_x, args.patch_size_y

# split up the response keys by comma in case there are multiple
args.response_key = args.response_key.split(',')

# read images in and convert to pytorch tensors
frames = read_frames(args.frame_dir, return_type = 'list')

# if n_frames is used, take the first n_frames frames
if args.n_frames: frames = frames[:args.n_frames]

# scale values to the range [0, 1]
frames = [np.float32(frame) / 255 for frame in frames]

# normalize each frame
frames = [(frame - np.mean(frame)) / (np.std(frame) + 1e-6) for frame in frames]

# get on and off frames
frames_off = frames.copy()
frames_off = [-frame for frame in frames_off]
frames = [frame * (frame > 0) for frame in frames]
frames_off = [frame * (frame > 0) for frame in frames_off]
frame_h, frame_w = frames[0].shape

# read response data and pupil location data
h5_responses = h5py.File(args.response_fpath, 'r+')
h5_pupil_locs = h5py.File(args.pupil_loc_fpath, 'r+')
keys = list(h5_responses.keys())

# filter out the keys based on the response_key
keys = [key for key in keys if all([filter in key for filter in args.response_key])]
print('[INFO] USING {} RESPONSE FILES.'.format(len(keys)))

# get all unique cell ids
cell_ids = list(set([key.split('_')[2] for key in keys]))

for cell_id in cell_ids:
    keys_cell = [key for key in keys if key.split('_')[2] == cell_id]
    patches_all_trials = np.zeros([0, psx * psy * args.n_frames_in_time * 2])
    responses_all_trials = np.zeros([0,])

    for key_cell in keys_cell:
        # read and format pupil locations
        pupil_locs_cell = h5_pupil_locs[key_cell][()]
        if args.n_frames: pupil_locs_cell = pupil_locs_cell[:args.n_frames, :]

        # read and format responses
        responses = h5_responses[key_cell][()]
        if args.n_frames: responses = responses[:args.n_frames]

        # find indices with valid pupil locations (not nan) for current and previous n_frames_in_time frames
        pupil_valid = [not val for val in np.isnan(pupil_locs_cell[:, 0])]
        inds_total = list(range(args.n_frames_in_time - 1, len(pupil_valid) - 1))
        inds_valid = [ind for ind in inds_total if all(pupil_valid[ind - (args.n_frames_in_time - 1):ind + 1])]

        # filter out pupil locations at the edge
        inds_valid = [ind for ind in inds_valid if (pupil_locs_cell[ind, 0] < (frame_w - psx // 2)) or (pupil_locs_cell[ind, 0] > psx // 2)]
        inds_valid = [ind for ind in inds_valid if (pupil_locs_cell[ind, 1] < (frame_h - psy // 2)) or (pupil_locs_cell[ind, 1] > psy // 2)]

        # just put in integers for the nans here, already know where the nans are
        x, y = pupil_locs_cell[:, 0], pupil_locs_cell[:, 1]
        x[x == np.nan] = 608 // 2
        y[y == np.nan] = 304 // 2
        x, y = np.int32(np.round(x)), np.int32(np.round(y))

        # get patches for all frames
        patches = [frame[r - psy // 2:r + psy // 2, c - psx // 2:c + psx // 2] for frame, r, c in zip(frames, y, x)]
        patches_off = [frame[r - psy // 2:r + psy // 2, c - psx // 2:c + psx // 2] for frame, r, c in zip(frames_off, y, x)]

        # add a third dimension to all frames so we can concatenate along that dim
        patches = [patch[..., None] for patch in patches]
        patches_off = [patch[..., None] for patch in patches_off]

        # aggregate patches in time
        patches = [np.concatenate(patches[ind - (args.n_frames_in_time - 1):ind + 1], axis = 2) for ind in inds_valid]
        patches_off = [np.concatenate(patches_off[ind - (args.n_frames_in_time - 1):ind + 1], axis = 2) for ind in inds_valid]

        # aggregate on and off and turn into array
        patches = [np.concatenate([patch, patch_off], axis = 2) for patch, patch_off in zip(patches, patches_off)]
        patches = np.asarray(patches)
        n, h, w, c = patches.shape

        # flatten each data sample and add to the rest of the patches for the cell from other trials
        patches = patches.reshape([n, -1])
        patches_all_trials = np.concatenate((patches_all_trials, patches), 0)

        # get responses for this trial and add to responses for other trials
        responses = responses[inds_valid]
        responses_all_trials = np.concatenate((responses_all_trials, responses))

    if args.downsample_training_set:
        patches_all_trials = patches_all_trials[::args.downsample_training_set]
        responses_all_trials = responses_all_trials[::args.downsample_training_set]

    # normalize inputs
    if args.feature_norm:
       patches_mean, patches_std = np.mean(patches_all_trials, 0), np.std(patches_all_trials, 0)
       patches_all_trials = (patches_all_trials - patches_mean) / (patches_std + 1e-6)

    assert(patches_all_trials.shape[0] == responses_all_trials.shape[0]), \
        'Number of training samples is not equal to the number of responses. Something went wrong.'
    print('[INFO] TRAINING ON {} SAMPLES.'.format(patches_all_trials.shape[0]))

    # create the alpha arguments and l1_ratio arguments
    alphas = 10 ** np.linspace(-6, 4, 75)
    l1_ratios = np.arange(0.1, 1.3, 0.3)

    # ElasticNet
    glm = ElasticNet(fit_intercept = False, max_iter = args.max_iters, selection = 'random')
    glm_cv = GridSearchCV(estimator = glm, param_grid = dict(alpha = alphas, l1_ratio = l1_ratios), cv = args.n_folds, n_jobs = args.n_jobs).fit(patches_all_trials, responses_all_trials)
    coefs = glm_cv.best_estimator_.coef_.reshape([args.patch_size_y, args.patch_size_x, args.n_frames_in_time * 2])
    print(glm_cv.best_estimator_.get_params())

    # write the coefficients to the h5file
    write_h5(args.save_path, str(cell_id), coefs)


h5_responses.close()
h5h5_pupil_locs.close()
