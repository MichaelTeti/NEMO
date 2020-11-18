from argparse import ArgumentParser
import os

import cv2
import numpy as np
import torch

from NEMO.utils.general_utils import (
    multiproc,
    add_string_to_fpaths,
    get_fpaths_in_dir,
    change_file_exts
)
from NEMO.utils.image_utils import spatial_whiten, max_min_scale


def whiten_imgs(old_and_new_fpaths, full_svd = False, scale_method = 'video'):
    '''
    Read in images based on fpaths, resize, and save in a new fpath.
    Args:
        old_and_new_fpaths (list of lists/tuples): List of (read_fpath, save_fpath) for each image.
        full_svd (bool): If True, use all SVD components.
    Returns:
        None
    '''

    for read_dir, save_dir in old_and_new_fpaths:
        for fpath_num, fpath in enumerate(read_dir):
            # read in the image and make grayscale
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

            # make a matrix to store the images if first image
            if fpath_num == 0:
                h, w = img.shape
                img_mat = np.zeros([len(read_dir), h * w])

            # add the image to the mat
            img_mat[fpath_num] = img.flatten()

        # do whitening
        img_mat_whitened = spatial_whiten(img_mat, full_matrix = full_svd)

        # scale frames based on max/min from entire video
        if scale_method == 'video':
            img_mat_whitened = max_min_scale(img_mat_whitened) * 255

        # save images
        for save_fpath_num, (new_fpath, flattened_img) in enumerate(zip(save_dir, img_mat_whitened)):
            # scale frame based on max/min from that frame if specified
            if scale_method == 'frame':
                flattened_img = max_min_scale(flattened_img)

            img_rec = flattened_img.reshape([h, w])
            if save_fpath_num == 0: os.makedirs(os.path.split(new_fpath)[0], exist_ok = True)
            cv2.imwrite(new_fpath, img_rec)



if __name__ == '__main__':
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
        default = 'frame',
        type = str,
        help = 'Whether to scale the frames by the min/max of the video or each individual frame.'
    )
    args = parser.parse_args()

    fpaths, save_fpaths = [], []
    for root, _, files in os.walk(args.data_dir_parent):
        if args.key and args.key in root:
            fpaths.append([os.path.join(root, file) for file in files])
            save_fpaths.append([os.path.join(root + '_whitened', os.path.splitext(file)[0] + '.png') for file in files])

    fpaths_and_save_fpaths = list(zip(fpaths, save_fpaths))
    whiten_imgs(old_and_new_fpaths = fpaths_and_save_fpaths, full_svd = args.full_svd)
