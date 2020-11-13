from argparse import ArgumentParser
import os, sys

from cv2 import imread, imwrite
import numpy as np

from NEMO.utils.general_utils import (
    multiproc,
    add_string_to_fpaths,
    get_fpaths_in_dir,
    change_file_exts
)
from NEMO.utils.image_utils import resize_and_keep_aspect


def resize_imgs(old_and_new_fpaths, desired_height, desired_width, aspect_ratio_tol):
    '''
    Read in images based on fpaths, resize, and save in a new fpath.
    Args:
        old_and_new_fpaths (list of lists/tuples): List of (read_fpath, save_fpath) for each image.
        desired_height (int): Height to resize each image.
        desired_width (int): Width to resize each image.
        aspect_ratio_tol (float): Discard images if absolute value between
            original aspect ratio and resized aspect ratio >= aspect_ratio_tol.
    Returns:
        None
    '''
    for fpath, new_fpath in old_and_new_fpaths:
        # read in the image
        img = imread(fpath)
        # calculate aspect ratio
        original_aspect = img.shape[1] / img.shape[0]
        # if aspect is not within aspect_ratio_tol of desired aspect, remove new dir then continue
        desired_aspect = desired_width / desired_height
        if abs(desired_aspect - original_aspect) > aspect_ratio_tol:
            if os.path.isdir(os.path.split(new_fpath)[0]): os.rmdir(os.path.split(new_fpath)[0])
            continue

        # resize the images
        img = resize_and_keep_aspect(img, desired_height, desired_width)
        # save the resized image
        imwrite(new_fpath, img)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'data_dir_parent',
        type = str,
        help = 'The parent directory with video frames and/or subdirs with video frames.'
    )
    parser.add_argument(
        'desired_height',
        type = int,
        default = 304
    )
    parser.add_argument(
        'desired_width',
        type = int,
        default = 608
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
        '--aspect_ratio_tol',
        type = float,
        default = 0.5,
        help = 'If actual aspect ratio is > aspect_ratio_tol, frames will not be resized.'
    )
    args = parser.parse_args()

    fpaths = get_fpaths_in_dir(args.data_dir_parent, key = args.key)
    save_fpaths = add_string_to_fpaths(fpaths, '_resized')
    save_fpaths = change_file_exts(save_fpaths, '.png')
    for fpath in save_fpaths: os.makedirs(os.path.split(fpath)[0], exist_ok = True)
    fpaths_and_save_fpaths = list(zip(fpaths, save_fpaths))
    multiproc(
        func = resize_imgs,
        iterator_key = 'old_and_new_fpaths',
        n_workers = args.n_workers,
        old_and_new_fpaths = fpaths_and_save_fpaths,
        desired_height = args.desired_height,
        desired_width = args.desired_width,
        aspect_ratio_tol = args.aspect_ratio_tol
    )
