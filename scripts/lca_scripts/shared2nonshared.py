from argparse import ArgumentParser
from glob import glob
import os

from oct2py import octave

parser = ArgumentParser()
parser.add_argument(
    'lca_ckpt_dir',
    type = str,
    help = 'The path to the LCA checkpoint directory where the weights are.'
)
parser.add_argument(
    'input_h',
    type = int,
    help = 'The height of the input images/frames.'
)
parser.add_argument(
    'input_w',
    type = int,
    help = 'The width of the input images/frames.'
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
args = parser.parse_args()

# make save_dir if doesn't exist
os.makedirs(args.save_dir, exist_ok = True)

# add OpenPV matlab utility directory to octave path
octave.addpath('/home/mteti/OpenPV/mlab/util')

# get a list of the filepaths
weight_fpaths = glob(os.path.join(args.lca_ckpt_dir, args.weight_file_key))
weight_fpaths.sort()

for frame_num, fpath in enumerate(weight_fpaths):
    pvp_data = octave.readpvpfile(fpath)
    weights = pvp_data[0]['values'][0]
    w_x, w_y, w_in, w_out = weights.shape