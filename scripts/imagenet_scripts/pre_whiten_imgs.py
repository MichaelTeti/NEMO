from argparse import ArgumentParser
import os

import cv2
import numpy as np

from NEMO.utils.general_utils import (
    multiproc,
    add_string_to_fpaths,
    get_fpaths_in_dir,
    change_file_exts,
)
from NEMO.utils.image_utils import max_min_scale, make_lgn_freq_filter


def pre_whiten_imgs(old_and_new_fpaths, f_0 = None):
    '''
    Read in images based on fpaths, whiten, and save in a new fpath.
    Args:
        old_and_new_fpaths (list of lists/tuples): List of (read_fpath, save_fpath) for each image.
        f_0 (int): Cycles/s desired.
    Returns:
        None
    '''

    for fpath_num, (fpath, new_fpath) in enumerate(old_and_new_fpaths):
        # read in the image and make grayscale
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[0], img.shape[1]

        # make the filter
        if fpath_num == 0:
            if not f_0: f_0 = np.ceil(min(h, w) * 0.4)
            ffilter = make_lgn_freq_filter(w, h, f_0 = f_0)

        # fft transform on image, filter, and go back to image
        img_fft = np.fft.fft2(img)
        img_fft *= ffilter
        img_rec = np.absolute(np.fft.ifft2(img_fft))

        # scale image to [0, 255] and write image
        img_scaled = max_min_scale(img_rec) * 255
        cv2.imwrite(new_fpath, img_scaled)


if __name__ == '__main__':
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
    fpaths_and_save_fpaths = list(zip(fpaths, save_fpaths))
    multiproc(
        func = pre_whiten_imgs,
        iterator_key = 'old_and_new_fpaths',
        n_workers = args.n_workers,
        old_and_new_fpaths = fpaths_and_save_fpaths,
        f_0 = args.f_0
    )
