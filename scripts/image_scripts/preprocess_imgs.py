from argparse import ArgumentParser
import os
import sys

from nemo.data.preprocess.image import (
    read_crop_write,
    read_downsample_write,
    read_resize_write,
    read_smooth_write,
    read_whiten_write
)
from nemo.data.utils import (
    add_string_to_fpaths,
    change_file_exts,
    search_files,
    multiproc
)

parser = ArgumentParser()
parser.add_argument(
    'data_dir_parent',
    type = str,
    help = 'The parent directory with video frames and/or subdirs with video frames.'
)
parser.add_argument(
    'op',
    type = str,
    choices = [
        'resize',
        'crop',
        'downsample',
        'smooth',
        'whiten'
    ]
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

# resize arguments
resize_args = parser.add_argument_group(description = 'Arguments for resize op.')
resize_args.add_argument(
    '--resize_h',
    type = int,
    default = 32,
    help = 'Desired height of the resized images.'
)
resize_args.add_argument(
    '--resize_w',
    type = int,
    default = 64,
    help = 'Desired width of the resized images.'
)
resize_args.add_argument(
    '--aspect_ratio_tol',
    type = float,
    default = 0.26,
    help = 'If actual aspect ratio is > aspect_ratio_tol, frames will not be resized.'
)

# crop arguments
crop_args = parser.add_argument_group(description = 'Arguments for crop op.')
crop_args.add_argument(
    '--crop_h',
    type = int,
    default = 32,
    help = 'Desired height to crop images to.'
)
crop_args.add_argument(
    '--crop_w',
    type = int,
    default = 64,
    help = 'Desired width to crop images to.'
)

# downsample arguments
downsample_args = parser.add_argument_group(description = 'Arguments for downsample op.')
downsample_args.add_argument(
    '--downsample_h',
    type = int,
    default = 2,
    help = 'The factor to downsample the height by.'
)
downsample_args.add_argument(
    '--downsample_w',
    type = int,
    default = 2,
    help = 'The factor to downsample the width by.'
)

# smoothing arguments
smoothing_args = parser.add_argument_group(description = 'Arguments for smoothing op.')
smoothing_args.add_argument(
    '--neighborhood',
    type = int,
    default = 9,
    help = 'Diameter of the pixel neighborhood.'
)
smoothing_args.add_argument(
    '--sigma_color',
    type = int,
    default = 75,
    help = 'Larger values mean larger differences in colors can be mixed together.'
)
smoothing_args.add_argument(
    '--sigma_space',
    type = int,
    default = 75,
    help = 'Larger values mean larger differences in space can be mixed together.'
)

# whitening arguments
whiten_args = parser.add_argument_group(description = 'Arguments for whitening op.')
whiten_args.add_argument(
    '--full_svd',
    action = 'store_true',
    help = 'If specified, use full SVD.'
)
whiten_args.add_argument(
    '--scale_method',
    type = str,
    choices = ['video', 'frame'],
    default = 'video',
    help = 'Whether to scale each frame from max/min of the video or each frame.'
)

args = parser.parse_args()


if args.op == 'whiten':

    fpaths, save_fpaths = [], []
    for root, _, files in os.walk(args.data_dir_parent):
        if files != []:
            if args.key and args.key not in root: continue 
            fpaths.append([os.path.join(root, file) for file in files])
            save_fpaths.append([
                os.path.join(
                    root + '_{}'.format(args.op), 
                    os.path.splitext(file)[0] + '.png'
                ) 
                for file in files
            ])

    read_whiten_write(
        read_fpaths = fpaths,
        write_fpaths = save_fpaths,
        full_svd = args.full_svd,
        scale_method = args.scale_method
    )

else: 

    fpaths = search_files(args.data_dir_parent, key = args.key)
    save_fpaths = add_string_to_fpaths(fpaths, '_{}'.format(args.op))
    save_fpaths = change_file_exts(save_fpaths, '.png')
    for fpath in save_fpaths: os.makedirs(os.path.split(fpath)[0], exist_ok = True)

    if args.op == 'resize':
        multiproc(
            func = read_resize_write,
            iterator_keys = ['read_fpaths', 'write_fpaths'],
            n_workers = args.n_workers,
            keep_list = True,
            read_fpaths = fpaths,
            write_fpaths = save_fpaths,
            desired_height = args.resize_h,
            desired_width = args.resize_w,
            aspect_ratio_tol = args.aspect_ratio_tol
        )
    elif args.op == 'crop':
        multiproc(
            read_crop_write,
            iterator_keys = ['read_fpaths', 'write_fpaths'],
            n_workers = args.n_workers,
            keep_list = True,
            read_fpaths = fpaths,
            write_fpaths = save_fpaths,
            crop_height = args.crop_h,
            crop_width = args.crop_w
        )
    elif args.op == 'downsample':
        multiproc(
            read_downsample_write,
            iterator_keys = ['read_fpaths', 'write_fpaths'],
            n_workers = args.n_workers,
            keep_list = True,
            read_fpaths = fpaths,
            write_fpaths = save_fpaths,
            downsample_h = args.downsample_h,
            downsample_w = args.downsample_w
        )
    elif args.op == 'smooth':
        multiproc(
            read_smooth_write,
            iterator_keys = ['read_fpaths', 'write_fpaths'],
            n_workers = args.n_workers,
            keep_list = True,
            read_fpaths = fpaths,
            write_fpaths = save_fpaths,
            neighborhood = args.neighborhood,
            sigma_color = args.sigma_color,
            sigma_space = args.sigma_space
        )