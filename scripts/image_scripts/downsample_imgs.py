from argparse import ArgumentParser
import os, sys

from cv2 import imread, imwrite
import numpy as np

from NEMO.utils.general_utils import (
    multiproc,
    get_fpaths_in_dir,
    add_string_to_fpaths,
    change_file_exts
)


def downsample_imgs(old_and_new_fpaths, downsample_h, downsample_w):
    '''
    Reads in images from fpaths, subsamples, and resaves them.
    
    Args:
        img_fpaths (list of lists/tuples): List of sublists/tuples, where each
            sublist is [read_path, save_path] for each image.
        downsample_h (int): The factor to downsample the image height by.
        downsample_w (int): The factor to downsample the image width by.
        
    Returns:
        None
    '''

    if type(downsample_h) != int or type(downsample_w) != int:
        raise TypeError('downsample_h/downsample_w should be ints, but are type {}/{}.'\
            .format(type(downsample_h), type(downsample_w)))

    for fpath, save_fpath in old_and_new_fpaths:
        # read in the video frames and downsample
        img = imread(fpath)[::downsample_h, ::downsample_w]
        
        # save the downsampled frame
        imwrite(save_fpath, img)


if __name__ == '__main__':
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
        downsample_imgs,
        iterator_key = 'old_and_new_fpaths',
        n_workers = args.n_workers,
        old_and_new_fpaths = fpaths_and_save_fpaths,
        downsample_h = args.downsample_h,
        downsample_w = args.downsample_w
    )
