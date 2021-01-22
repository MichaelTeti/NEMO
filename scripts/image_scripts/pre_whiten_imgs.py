from argparse import ArgumentParser
import os

import cv2
import numpy as np

from nemo.data.preprocess import max_min_scale, make_lgn_freq_filter, read_pre_whiten_write
from nemo.data.utils import (
    multiproc,
    add_string_to_fpaths,
    get_fpaths_in_dir,
    change_file_exts,
)


parser = ArgumentParser()
parser.add_argument(
    'data_dir_parent',
    type = str,
    help = 'The parent directory with video frames and/or subdirs with video frames.'
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
parser.add_argument(
    '--f_0',
    type = int,
    help = 'The number of cycles per picture. If not given, it will be calculated.'
)
args = parser.parse_args()

fpaths = get_fpaths_in_dir(args.data_dir_parent, key = args.key)
save_fpaths = add_string_to_fpaths(fpaths, '_pre-whitened')
save_fpaths = change_file_exts(save_fpaths, '.png')
for fpath in save_fpaths: os.makedirs(os.path.split(fpath)[0], exist_ok = True)
multiproc(
    func = read_pre_whiten_write,
    iterator_keys = ['read_fpaths', 'write_fpaths'],
    n_workers = args.n_workers,
    read_fpaths = fpaths,
    write_fpaths = save_fpaths,
    f_0 = args.f_0
)
