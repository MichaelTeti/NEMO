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
from NEMO.utils.image_utils import center_crop


def crop_imgs(old_and_new_fpaths, crop_height, crop_width):
    '''
    Read in images, crop them, and resave them.
    Args:
        old_and_new_fpaths (list of lists/tuples): List of (read_fpath, save_fpath) for each image.
        crop_height (int): Height of the cropped image.
        crop_width (int): Width of the cropped image.
    Returns:
        None
    '''

    if type(crop_height) != int or type(crop_width) != int:
        raise TypeError('crop_height/crop_width should be ints but are of type {}/{}.'\
            .format(type(crop_height), type(crop_width)))

    for fpath, save_fpath in old_and_new_fpaths:
        # read in the image
        img = imread(fpath)
        # check if the image is smaller than the specified crop dims
        if img.shape[0] < crop_height or img.shape[1] < crop_width:
            if os.path.isdir(os.path.split(save_fpath)[0]): os.rmdir(os.path.split(save_fpath)[0])
            continue

        # crop it and save
        img = center_crop(img, crop_height, crop_width)
        imwrite(save_fpath, img)


if __name__ == '__main__':
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
        func = crop_imgs,
        iterator_key = 'old_and_new_fpaths',
        n_workers = args.n_workers,
        old_and_new_fpaths = fpaths_and_save_fpaths,
        crop_height = args.crop_height,
        crop_width = args.crop_width
    )
