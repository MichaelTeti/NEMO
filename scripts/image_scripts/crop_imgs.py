from argparse import ArgumentParser
import os, sys

import numpy as np

from nemo.data.preprocess import read_crop_write
from nemo.data.utils import (
    multiproc,
    get_fpaths_in_dir,
    add_string_to_fpaths,
    change_file_exts
)


parser = ArgumentParser()
parser.add_argument(
    'data_dir_parent',
    type = str
)
parser.add_argument(
    'crop_height',
    type = int,
    default = 256
)
parser.add_argument(
    'crop_width',
    type = int,
    default = 256
)
parser.add_argument(
    '--n_workers',
    type = int,
    default = 8
)
parser.add_argument(
    '--key',
    type = str
)
args = parser.parse_args()

fpaths = get_fpaths_in_dir(args.data_dir_parent, key = args.key)
save_fpaths = add_string_to_fpaths(fpaths, '_cropped')
save_fpaths = change_file_exts(save_fpaths, '.png')
for fpath in save_fpaths: os.makedirs(os.path.split(fpath)[0], exist_ok = True)
fpaths_and_save_fpaths = list(zip(fpaths, save_fpaths))
multiproc(
    func = read_crop_write,
    iterator_keys = ['read_fpaths', 'write_fpaths'],
    n_workers = args.n_workers,
    read_fpaths = fpaths,
    write_fpaths = save_fpaths,
    crop_height = args.crop_height,
    crop_width = args.crop_width
)
