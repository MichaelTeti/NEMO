from argparse import ArgumentParser
from glob import glob
import os

import numpy as np
from oct2py import octave
from progressbar import ProgressBar

parser = ArgumentParser()
parser.add_argument(
    'lca_ckpt_dir',
    type = str,
    help = 'The path to the LCA checkpoint directory where the weights are.'
)
parser.add_argument(
    'desired_patch_h',
    type = int,
    help = 'The desired height of the patches.'
)
parser.add_argument(
    'desired_patch_w',
    type = int,
    help = 'The desired width of the patches.'
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
    help = 'Path to the <model_layer>_A.pvp file in the ckpt_dir (only needed) if \
        n_features_keep is specified.'
)
parser.add_argument(
    '--openpv_path',
    type = str,
    default = '/home/mteti/OpenPV/mlab/util',
    help = 'Path to the OpenPv/mlab/util directory.'
)
args = parser.parse_args()

# make save_dir if doesn't exist
os.makedirs(args.save_dir, exist_ok = True)

# add OpenPV matlab utility directory to octave path
octave.addpath(args.openpv_path)

# get a list of the filepaths
weight_fpaths = glob(os.path.join(args.lca_ckpt_dir, args.weight_file_key))
weight_fpaths.sort()

# get top features if specified
if args.n_features_keep:
    act_data = octave.readpvpfile(args.act_fpath)
    n_batch = len(act_data)
    acts = np.concatenate([act_data[b]['values'][None, ...] for b in range(n_batch)], 0)
    assert args.n_features_keep <= acts.shape[-1]
    mean_acts = np.mean(acts, (0, 1, 2))
    inds = list(range(mean_acts.size))
    sorted_inds = [ind for _, ind in sorted(zip(mean_acts, inds), reverse = True)]
    sorted_inds_keep = sorted_inds[:args.n_features_keep]

for frame_num, fpath in ProgressBar()(enumerate(weight_fpaths)):
    # read in the data from the weight file
    feat_data = octave.readpvpfile(fpath)
    weights = feat_data[0]['values'][0]
    
    # pull out top features if specified
    if args.n_features_keep: 
        weights = weights[..., sorted_inds_keep]
    
    w_x, w_y, w_in, w_out = weights.shape
    assert args.desired_patch_w >= w_x
    assert args.desired_patch_h >= w_y
    
    # compute the new number of features you will have
    nx = args.desired_patch_w - (w_x - 1)
    ny = args.desired_patch_h - (w_y - 1)
    w_out_new = nx * ny * w_out
    nonshared = np.zeros(
        [
            args.desired_patch_w, 
            args.desired_patch_h, 
            w_in, 
            w_out_new
        ], 
        dtype = np.float64
    )
    
    count = 0
    for k in range(w_out):
        for i in range(nx):
            for j in range(ny):
                nonshared[i:i + w_x, j:j + w_y, :, count] = weights[:, :, :, k]
                count += 1
                
    # write the new features
    write_fpath = os.path.join(args.save_dir, os.path.split(fpath)[1])
    feat_data[0]['values'][0] = nonshared
    octave.push(['write_fpath', 'feat_data'], [write_fpath, feat_data])
    octave.eval('writepvpsharedweightfile(write_fpath, feat_data)')
    
print('[INFO] NONSHARED GRID SIZE IS {}x{}x{}.'.format(ny, nx, w_out_new))
