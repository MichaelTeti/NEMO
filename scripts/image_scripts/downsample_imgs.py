from argparse import ArgumentParser
import os, sys

import numpy as np

from nemo.data.preprocess import read_downsample_write
from nemo.data.utils import (
    multiproc,
    get_fpaths_in_dir,
    add_string_to_fpaths,
    change_file_exts
)


parser = ArgumentParser()
parser.add_argument(
    'data_dir_parent',
    type = str,
    help = 'Directory to recurse through to find images.'
)
parser.add_argument(
    'downsample_h',
    type = int,
    help = 'Factor to downsample by. e.g. if the value is 3, it will take \
            every third pixel value along the row dim.'
)
parser.add_argument(
    'downsample_w',
    type = int,
    help = 'Factor to downsample by. e.g. if the value is 3, it will take \
            every third pixel value along the col dim.'
)
parser.add_argument(
    '--n_workers',
    type = int,
    default = 2,
    help = 'Number of CPUs to use. Default is 2.'
)
parser.add_argument(
    '--key',
    type = str,
    help = 'A key phrase used to filter out desired folders / images. Default is None.'
)
args = parser.parse_args()

fpaths = get_fpaths_in_dir(args.data_dir_parent, key = args.key)
save_fpaths = add_string_to_fpaths(fpaths, '_downsampled')
save_fpaths = change_file_exts(save_fpaths, '.png')
for fpath in save_fpaths: os.makedirs(os.path.split(fpath)[0], exist_ok = True)
fpaths_and_save_fpaths = list(zip(fpaths, save_fpaths))
multiproc(
    read_downsample_write,
    iterator_keys = ['read_fpaths', 'write_fpaths'],
    n_workers = args.n_workers,
    read_fpaths = fpaths,
    write_fpaths = save_fpaths,
    downsample_h = args.downsample_h,
    downsample_w = args.downsample_w
)
