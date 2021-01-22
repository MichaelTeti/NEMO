from argparse import ArgumentParser
import os

import cv2
import numpy as np
import torch

from nemo.data.preprocess import read_whiten_write
from nemo.data.utils import (
    multiproc,
    add_string_to_fpaths,
    get_fpaths_in_dir,
    change_file_exts
)


parser = ArgumentParser()
parser.add_argument(
    'data_dir_parent',
    type = str,
    help = 'The parent directory with video frames and/or subdirs with video frames.'
)
parser.add_argument(
    '--key',
    type = str,
    help = 'If key is specified and not in path of a frame, that frame will not be used.'
)
parser.add_argument(
    '--full_svd',
    action = 'store_true',
    help = 'If specified, use all SVD components.'
)
parser.add_argument(
    '--scale_method',
    choices = ['video', 'frame'],
    default = 'video',
    type = str,
    help = 'Whether to scale the frames by the min/max of the video or each individual frame.'
)
args = parser.parse_args()

fpaths, save_fpaths = [], []
for root, _, files in os.walk(args.data_dir_parent):
    if args.key and args.key not in root: continue 
    fpaths.append([os.path.join(root, file) for file in files])
    save_fpaths.append([os.path.join(root + '_whitened', os.path.splitext(file)[0] + '.png') for file in files])

read_whiten_write(
    read_fpaths = fpaths,
    write_fpaths = save_fpaths,
    full_svd = args.full_svd,
    scale_method = args.scale_method
)
