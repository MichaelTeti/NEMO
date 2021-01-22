from argparse import ArgumentParser
import os, sys

import cv2
import numpy as np

from nemo.data.preprocess import read_smooth_write
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
    '--neighborhood',
    type = int,
    default = 5,
    help = 'Diameter of each pixel neighborhood that is used during filtering. Should be odd.'
)
parser.add_argument(
    '--sigma_color',
    type = float,
    default = 50,
    help = 'Filter sigma in the color space. A larger value of the parameter means that \
        farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together.'
)
parser.add_argument(
    '--sigma_space',
    type = float,
    default = 50,
    help = 'Filter sigma in the coordinate space. A larger value of the parameter means that \
        farther pixels will influence each other as long as their colors are close enough.'
)
parser.add_argument(
    '--n_workers',
    type = int,
    default = 8
)
parser.add_argument(
    '--key',
    type = str,
    help = 'If key is specified and not in path of a frame, that frame will not be used.'
)
args = parser.parse_args()

fpaths = get_fpaths_in_dir(args.data_dir_parent, key = args.key)
save_fpaths = add_string_to_fpaths(fpaths, '_smoothed')
save_fpaths = change_file_exts(save_fpaths, '.png')
for fpath in save_fpaths: os.makedirs(os.path.split(fpath)[0], exist_ok = True)
multiproc(
    func = read_smooth_write,
    iterator_keys = ['read_fpaths', 'write_fpaths'],
    n_workers = args.n_workers,
    read_fpaths = fpaths,
    write_fpaths = save_fpaths,
    neighborhood = args.neighborhood,
    sigma_color = args.sigma_color,
    sigma_space = args.sigma_space
)
