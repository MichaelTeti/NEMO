from argparse import ArgumentParser
import os

from glob import glob
import imageio
import numpy as np
from oct2py import octave

from NEMO.utils.image_utils import max_min_scale

parser = ArgumentParser()
parser.add_argument(
    'ckpt_dir',
    type = str,
    help = 'Path to the directory containing the _W.pvp files to view.'
)
parser.add_argument(
    'save_dir',
    type = str,
    help = 'Path to the directory where images will be written out.'
)
parser.add_argument(
    'n_features_y',
    type = int,
    help = 'Height of the grid.'
)
parser.add_argument(
    'n_features_x',
    type = int,
    help = 'Width of the grid.'
)
parser.add_argument(
    '--weight_file_key',
    type = str,
    default = 'S1ToFrame*ReconError_W.pvp',
    help = 'A key to select out the specific weight files to plot in ckpt_dir.'
)
parser.add_argument(
    '--openpv_path',
    type = str,
    default = '/home/mteti/OpenPV/mlab/util',
    help = 'Path to the OpenPV matlab utilities directory.'
)
args = parser.parse_args()

# make sure save_dir exists or create it
os.makedirs(args.save_dir, exist_ok = True)

# add OpenPV matlab utility directory to octave path
octave.addpath(args.openpv_path)

# get the weight files
feat_fpaths = glob(os.path.join(args.ckpt_dir, args.weight_file_key))
feat_fpaths.sort()
n_frames = len(feat_fpaths)

for frame_num, feat_fpath in enumerate(feat_fpaths):
    # read in the features and reshape 
    feat_data = octave.readpvpfile(feat_fpath)
    weights = feat_data[0]['values'][0]
    n_feats = weights.shape[-1] // args.n_features_x // args.n_features_y
    w_x, w_y, w_in, _ = weights.shape
    weights = weights.reshape([w_x, w_y, w_in, n_feats, args.n_features_x, args.n_features_y])
    weights = weights.transpose([1, 0, 2, 3, 5, 4]) # go from x, y to y, x for images
    
    # compile all frames here
    if frame_num == 0:
        weights_all_frames = np.zeros([n_frames] + list(weights.shape))
        
    # add weights from current frame
    weights_all_frames[frame_num] = weights
    
# scale all features to [0, 255]
weights_all_frames = max_min_scale(weights_all_frames) * 255

# add features to a 2D (or 3D for color) grid and write out 
for feat_num in range(n_feats):
    feat_grid = np.zeros([n_frames, args.n_features_y * w_y, args.n_features_x * w_x, w_in])
    
    for i in range(args.n_features_y):
        for j in range(args.n_features_x):
            feat_grid[:, i*w_y:(i+1)*w_y, j*w_x:(j+1)*w_x, :] = weights_all_frames[..., feat_num, i, j]
    
    # put a black border in between patches
    feat_grid[:, ::w_y, :, :] = np.amin(feat_grid)
    feat_grid[:, :, ::w_x, :] = np.amin(feat_grid)
    
    # get rid of single channel if grayscale and write
    feat_grid = np.uint8(np.squeeze(feat_grid))
    imageio.mimwrite(
        os.path.join(args.save_dir, 'feature{}.gif'.format(feat_num)),
        [feat_grid[frame_num] for frame_num in range(n_frames)]
    )
