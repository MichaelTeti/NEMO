from argparse import ArgumentParser
from glob import glob
import logging
import os

import numpy as np
from oct2py import octave
from progressbar import ProgressBar

from nemo.model.analysis.lca import get_mean_activations

parser = ArgumentParser()
parser.add_argument(
    'lca_ckpt_dir',
    type = str,
    help = 'The path to the LCA checkpoint directory where the weights are.'
)
parser.add_argument(
    'input_h',
    type = int,
    help = 'Height of the input video frames / images.'
)
parser.add_argument(
    'input_w',
    type = int,
    help = 'Width of the input video frames / images.'
)
parser.add_argument(
    '--stride_x',
    type = int,
    default = 1,
    help = 'Stride of the original patches inside the input in the x dim.'
)
parser.add_argument(
    '--stride_y',
    type = int,
    default = 1,
    help = 'Stride of the original patches inside the input in the y dim.'
)
parser.add_argument(
    '--weight_file_key',
    type = str,
    default = 'S1ToFrame*ReconError_W.pvp',
    help = 'A key to help select out the desired weight files in the ckpt_dir.'
)
parser.add_argument(
    '--save_dir',
    type = str,
    default = 'NonsharedWeights',
    help = 'The directory to save the outputs of this script in.'
)
parser.add_argument(
    '--n_features_keep',
    type = int,
    help = 'How many features to keep.'
)
parser.add_argument(
    '--act_fpath',
    type = str,
    help = 'Path to the <model_layer>_A.pvp file in the ckpt_dir (only needed if \
        n_features_keep is specified).'
)
parser.add_argument(
    '--openpv_path',
    type = str,
    default = '/home/mteti/OpenPV/mlab/util',
    help = 'Path to the OpenPv/mlab/util directory.'
)
args = parser.parse_args()

logging.basicConfig(
    format='%(asctime)s -- %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p', 
    level = logging.INFO
)

# make save_dir if doesn't exist
os.makedirs(args.save_dir, exist_ok = True)

# add path to OpenPV octave utils
octave.addpath(args.openpv_path)

# get a list of the filepaths
weight_fpaths = glob(os.path.join(args.lca_ckpt_dir, args.weight_file_key))
weight_fpaths.sort()

# get top features if specified
if args.n_features_keep:
    _, _, sorted_inds = get_mean_activations(args.act_fpath, openpv_path = args.openpv_path)
    sorted_inds_keep = sorted_inds[:args.n_features_keep]

for frame_num, fpath in ProgressBar()(enumerate(weight_fpaths)):
    # read in the data from the weight file
    feat_data = octave.readpvpfile(fpath)
    weights = feat_data[0]['values'][0]
    
    # pull out top features if specified
    if args.n_features_keep: 
        weights = weights[..., sorted_inds_keep]
    
    # get the shape of the weights and make sure they satisfy certain conditions
    w_x, w_y, w_in, w_out = weights.shape
    
    # compute the new number of features you will have and make placeholder to store them
    nx = args.input_w - (w_x - 1)
    ny = args.input_h - (w_y - 1)
    w_out_new = (nx // args.stride_x) * (ny // args.stride_y) * w_out
    nonshared = np.zeros([args.input_w, args.input_h, w_in, w_out_new], dtype = np.float64)

    # fill in the original features in the simple cell tensor
    count = 0
    for k in range(w_out):
        for i in range(0, nx, args.stride_x):
            for j in range(0, ny, args.stride_y):
                nonshared[i:i + w_x, j:j + w_y, :, count] = weights[:, :, :, k]
                count += 1
                
    # write the new features
    write_fpath = os.path.join(args.save_dir, os.path.split(fpath)[1])
    feat_data[0]['values'][0] = nonshared
    octave.push(['write_fpath', 'feat_data'], [write_fpath, feat_data])
    octave.eval('writepvpsharedweightfile(write_fpath, feat_data)')
    
logging.info('NONSHARED GRID SIZE IS {}x{}x{}.'.format(
    ny // args.stride_y, 
    nx // args.stride_x, 
    w_out
))
