from argparse import ArgumentParser
import os

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from progressbar import ProgressBar
from sklearn.linear_model import ElasticNetCV as ElasticNet

from NEMO.utils.image_utils import read_frames, max_min_scale
from NEMO.utils.model_utils import create_temporal_design_mat, save_args


parser = ArgumentParser()
parser.add_argument(
    'trace_dir',
    type = str,
    help = 'Path containing the .txt files with trial-averaged traces \
        for a single session type and stimulus.'
)
parser.add_argument(
    'stimulus_dir',
    type = str,
    help = 'Path to the stimulus templates corresponding to the trace_dir.'
)
parser.add_argument(
    'save_dir',
    type = str,
    help = 'Directory to save results in (if mode = train) or to read in strfs \
        from and write test results to (if mode = test).' 
)
parser.add_argument(
    '--write_rf_images',
    action = 'store_true',
    help = 'If specified, write the receptive fields out as images/.gifs.'
)
parser.add_argument(
    '--write_mse_plots',
    action = 'store_true',
    help = 'If specified, write the mse grid path as a plot.'
)
parser.add_argument(
    '--mode',
    type = str,
    choices = ['train', 'test'],
    default = 'train',
    help = 'Whether to train a new model or test a trained model.'
)


model_args = parser.add_argument_group(
    'model parameters',
    description = 'Parameter settings for the model and cross-validation.'
)
model_args.add_argument(
    '--n_frames_in_time',
    type = int,
    default = 9,
    help = 'The number of consecutive video frames to comprise a single input.'
)
model_args.add_argument(
    '--n_jobs',
    type = int,
    default = 8,
    help = 'Number of jobs for the model.'
)
model_args.add_argument(
    '--max_iter',
    type = int,
    default = 5000,
    help = 'The maximum number of iterations for the model.'
)
model_args.add_argument(
    '--n_alphas',
    type = int,
    default = 75,
    help = 'The number of alpha (aka lambda) values to search over.'
)
model_args.add_argument(
    '--min_l1_ratio',
    type = float,
    default = 1e-6, # must be non-zero for some reason 
    help = 'The minimum l1_ratio to try in the grid search.'
)
model_args.add_argument(
    '--max_l1_ratio',
    type = float,
    default = 1.0,
    help = 'The maximum l1_ratio to try in the grid search.'
)
model_args.add_argument(
    '--n_l1_ratios',
    type = int,
    default = 6,
    help = 'The number of l1_ratios to try in the range [min_l1_ratio, max_l1_ratio].'
)

args = parser.parse_args()

# make sure the save_dir exists or create it
if args.mode == 'train':
    os.makedirs(args.save_dir, exist_ok = True)
else:
    assert os.path.isdir(args.save_dir)

# save the args in the save_dir
if args.mode == 'train': save_args(args, args.save_dir)

# get a list of all cell traces in trace_dir
trace_fpaths = [os.path.join(args.trace_dir, f) for f in os.listdir(args.trace_dir)]

assert(all([os.path.splitext(fpath)[1] == '.txt' for fpath in trace_fpaths])), \
    'All files in trace_dir must be .txt files made by the get_trial_averaged_responses.py script.'

# get array of stimulus frames
stimulus = read_frames(args.stimulus_dir, gray = True)
_, h, w = stimulus.shape

# create the temporal design mat
design_mat = create_temporal_design_mat(stimulus, n_frames_in_time = args.n_frames_in_time)
n_samples = design_mat.shape[0]
design_mat = design_mat.reshape([n_samples, -1])

# standardize the columns of the design matrix
mean_vec = np.mean(design_mat, 0)
std_vec = np.std(design_mat, 0)
design_mat = (design_mat - mean_vec) / std_vec

def train(X, trace_fpaths, save_dir, min_l1_ratio = 1e-6, max_l1_ratio = 0.6, n_l1_ratios = 4, 
          n_frames_in_time = 9, n_alphas = 50, max_iter = 5000, n_jobs = 4, 
          write_mse_plots = False, write_rf_images = False):
          
    # check if save dirs they exist / create them if they don't
    if write_rf_images:
        rf_img_dir = os.path.join(save_dir, 'ReceptiveFieldImages')
        os.makedirs(rf_img_dir, exist_ok = True)

    if write_mse_plots:
        mse_plot_dir = os.path.join(save_dir, 'MSEPathPlots')
        os.makedirs(mse_plot_dir, exist_ok = True)

    # create list of l1_ratios to try
    l1_ratios = np.linspace(min_l1_ratio, max_l1_ratio, n_l1_ratios)
    print('[INFO] L1 RATIOS: {}'.format(l1_ratios))

    for fpath in ProgressBar()(trace_fpaths):
        # pull cell ID from filename for later saving of results
        cell_id = os.path.splitext(os.path.split(fpath)[1])[0]

        # read in the traces 
        traces = pd.read_csv(fpath)
    
        # cut the traces to make up for edge effects when compiling input video frame sequences
        traces = traces.iloc[:, n_frames_in_time - 1:].to_numpy()[0]

        # center traces so we don't have to fit an intercept
        traces -= np.mean(traces)    

        # initialize model
        elastic_net = ElasticNet(
            l1_ratio = l1_ratios,
            n_alphas = n_alphas,
            fit_intercept = False, # traces are centered
            selection = 'random',  # coeff update order for coordinate descent
            normalize = False,
            max_iter = max_iter,
            cv = 5,
            n_jobs = n_jobs
        )

        # fit model and get params
        elastic_net.fit(design_mat, traces)
        strf = elastic_net.coef_

        # reshape the strs back to height x width
        strf = strf.reshape([h, w, n_frames_in_time])

        # get the mse across folds, alphas, and lambdas
        if write_mse_plots:
            mse_path = elastic_net.mse_path_
            alphas = elastic_net.alphas_
            
            for i, l1_ratio in enumerate(l1_ratios):
                mu = np.mean(mse_path[i], 1)
                se = np.std(mse_path[i], 1) / np.sqrt(mse_path.shape[-1])
                plt.errorbar(
                    alphas[i],
                    mu,
                    yerr = se,
                    label = str(l1_ratio)
                )
            plt.xlabel('Log Lambda')
            plt.ylabel('CV MSE')
            plt.xscale('log')
            plt.legend(title = 'alpha')
            plt.savefig(
                os.path.join(mse_plot_dir, cell_id + '.png'),
                bbox_inches = 'tight'
            )
            plt.close() 
    
        # save the rf in an .h5 file where all will be saved
        with h5py.File(os.path.join(save_dir, 'strfs.h5'), 'a') as f:
            if cell_id not in list(f.keys()):
                f.create_dataset(cell_id, data = strf)    

        # write out the image if specified
        if write_rf_images:
            strf = np.uint8(max_min_scale(strf) * 255)
            imageio.mimwrite(
                os.path.join(rf_img_dir, cell_id + '.gif'),
                [strf[..., i] for i in range(n_frames_in_time)]
            )
