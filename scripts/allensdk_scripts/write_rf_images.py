from argparse import ArgumentParser
import os

import cv2
import h5py
import numpy as np

from nemo.data.preprocess.image import max_min_scale


parser = ArgumentParser()
parser.add_argument(
    'rf_fpath',
    type = str,
    help = 'Path to the receptive lsn_rfs.h5 file.'
)
parser.add_argument(
    'write_dir',
    type = str,
    help = 'Directory to save the rf images in.'
)
parser.add_argument(
    '--height',
    type = int,
    default = 64,
    help = 'Height to resize the rf to.'
)
parser.add_argument(
    '--width',
    type = int,
    default = 112,
    help = 'Width to resize the rf to.'
)
args = parser.parse_args()

on_dir = os.path.join(args.write_dir, 'on')
off_dir = os.path.join(args.write_dir, 'off')
os.makedirs(on_dir, exist_ok = True)
os.makedirs(off_dir, exist_ok = True)

with h5py.File(args.rf_fpath, 'r') as h5file:
    cell_ids = list(h5file.keys())

    for cell_id in cell_ids:
        rf = h5file[cell_id][()]
        rf[np.isnan(rf)] = 0
        
        on = np.uint8(max_min_scale(rf[..., 0]) * 255)
        off = np.uint8(max_min_scale(rf[..., 1]) * 255)
        
        on = cv2.resize(on, (args.width, args.height))
        off = cv2.resize(off, (args.width, args.height))
        
        cv2.imwrite(os.path.join(on_dir, '{}.png'.format(cell_id)), on)
        cv2.imwrite(os.path.join(off_dir, '{}.png'.format(cell_id)), off)