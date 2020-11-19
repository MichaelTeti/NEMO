from argparse import ArgumentParser
import os, sys

import cv2
import numpy as np

from NEMO.utils.general_utils import (
    multiproc,
    add_string_to_fpaths,
    get_fpaths_in_dir,
    change_file_exts
)
from NEMO.utils.image_utils import resize_and_keep_aspect


def blur_imgs(old_and_new_fpaths, neighborhood = 9, sigma_color = 75, sigma_space = 75):
    '''
    Read in images based on fpaths, resize, and save in a new fpath.

    Args:
        old_and_new_fpaths (list of lists/tuples): List of (read_fpath, save_fpath) for each image.
        neighborhood (int): Diameter of the pixel neighborhood.
        sigma_color (float): Larger values mean larger differences in colors can be mixed together.
        sigma_space (float): Larger values mean larger differences in space can be mixed together.

    Returns:
        None
    '''

    for fpath, new_fpath in old_and_new_fpaths:
        # read in the image
        img = cv2.imread(fpath)

        # blur the image
        img = cv2.bilateralFilter(
            img,
            d = neighborhood,
            sigmaColor = sigma_color,
            sigmaSpace = sigma_space
        )

        # save the resized image
        cv2.imwrite(new_fpath, img)



if __name__ == '__main__':
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
    save_fpaths = add_string_to_fpaths(fpaths, '_blurred')
    save_fpaths = change_file_exts(save_fpaths, '.png')
    for fpath in save_fpaths: os.makedirs(os.path.split(fpath)[0], exist_ok = True)
    fpaths_and_save_fpaths = list(zip(fpaths, save_fpaths))
    multiproc(
        func = blur_imgs,
        iterator_key = 'old_and_new_fpaths',
        n_workers = args.n_workers,
        old_and_new_fpaths = fpaths_and_save_fpaths,
        neighborhood = args.neighborhood,
        sigma_color = args.sigma_color,
        sigma_space = args.sigma_space
    )
