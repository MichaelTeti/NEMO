import os
from random import randint

from cv2 import (
    imread,
    imwrite,
    resize
)
import numpy as np


def resize_and_keep_aspect(img, desired_height, desired_width):
    '''
    Resizes an image to a new size while keeping the original aspect ratio by
    cropping the larger side.
    Args:
        img (np.ndarray): Color or grayscale image to resize.
        desired_height (int): Height to resize img to.
        desired_width (int): Width to resize image to.
    Returns:
        img_resized (np.ndarray): The resized image with the same aspect ratio as img.
    '''

    if type(desired_height) != int or type(desired_width) != int:
        raise TypeError('desired_height and desired_width should be of type int, \
                         but are of types {} / {} instead.'.format(type(desired_height), type(desired_width)))

    if desired_height <= 0 or desired_width <= 0:
            raise ValueError('desired_height and desired_width should be > 0, but \
                              have values {} / {}.'.format(desired_height, desired_width))

    h, w = img.shape[:2]
    desired_aspect = desired_width / desired_height
    actual_aspect = w / h
    if desired_aspect > actual_aspect:
        crop_height = h - int(w * desired_height / desired_width)
        if crop_height == 1:
            img = img[:-1]
        elif crop_height % 2 != 0:
            img = img[crop_height // 2:h - (crop_height // 2 + 1)]
        else:
            img = img[crop_height // 2: h - crop_height // 2]
    else:
        crop_width = w - int(h * desired_width / desired_height)
        if crop_width == 1:
            img = img[:, :-1]
        elif crop_width % 2 != 0:
            img = img[:, crop_width // 2:w - (crop_width // 2 + 1)]
        else:
            img = img[:, crop_width // 2:w - crop_width // 2]

    img_resized = resize(img, (desired_width, desired_height))
    return img_resized


def center_crop(img, crop_height, crop_width):
    '''
    Takes a center crop from an image given crop height and width.
    Args:
        img (np.ndarray): Image to crop with either 1, 3, or 4 channels.
        crop_height (int): Height of the center crop.
        crop_width (int): Width of the center crop.
    Returns:
        img_cropped (np.ndarray): The cropped image.
    '''

    acceptable_types = [int, np.int16, np.uint8, np.int32, np.int64]
    if type(crop_height) not in acceptable_types or type(crop_width) not in acceptable_types:
        raise TypeError('crop_height and crop_width should be of type int, \
                         but are of types {} / {} instead.'.format(type(crop_height), type(crop_width)))

    if crop_height <= 0 or crop_width <= 0:
            raise ValueError('crop_height and crop_width should be > 0, but \
                              have values {} / {}.'.format(crop_height, crop_width))

    crop_y = img.shape[0] // 2 - crop_height // 2
    crop_x = img.shape[1] // 2 - crop_width // 2
    img_cropped = img[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

    return img_cropped


def get_img_frame_names(n_frames):
    '''
    Generate numbers up to n_frames with leading zeros (useful for saving video array as images).
    Args:
        n_frames (int): The number of frames in the array (i.e. number to count up to).
    Returns:
        frame_numbers (list of strings)
    '''

    if type(n_frames) not in [int, np.uint8, np.int16, np.int32, np.int64]:
        raise TypeError('Arg n_frames should be of type int, but is of type {}.'.format(type(n_frames)))

    n_decimals = len(str(n_frames))
    return ['0' * (n_decimals - len(str(i))) + str(i) for i in range(n_frames)]



def save_vid_array_as_frames(vid_arrays_and_save_dirs):
    '''
    Save a video represented as an array as individual frames.
    Args:
        vid_array_and_save_dir (list): A list of 2-tuples / lists, where each
            tuple is composed of a NxHxWxC (or NxHxW for grayscale) array and an
            fpath to save the array at.
    Returns:
        None
    '''

    for vid_array, save_dir in vid_arrays_and_save_dirs:
        os.makedirs(save_dir, exist_ok = True)
        n_frames = vid_array.shape[0]
        fnames = [fname + '.png' for fname in get_img_frame_names(n_frames)]
        fpaths = [os.path.join(save_dir, fname) for fname in fnames]
        for i_frame, (frame, fpath) in enumerate(zip(vid_array, fpaths)):
            imwrite(fpath, np.uint8(frame))


def read_frames(dir, return_type = 'array'):
    '''
    Traverse a directory structure, reading in all images along the way and returning as list or np.ndarray.
    Args:
        dir (str): Directory to start at.
        return_type ('array' or 'list'): Data structure to return the video frames as.
    Returns:
        Video frames.
    '''

    frames = []
    for root, dirs, files in os.walk(dir):
        files.sort()
        for file in files:
            if file.split('.')[1] in ['jpeg', 'jpg', 'JPG', 'JPEG', 'PNG', 'png']:
                frames.append(imread(os.path.join(root, file)))
            else:
                raise NotImplementedError('read_frames only implemented for JPEG or PNG images.')

    if return_type == 'array':
        return np.array(frames)
    elif return_type == 'list':
        return frames


def spatial_whiten(imgs, return_zca_mat = False, full_matrix = False):
    '''
    Function to ZCA whiten images.
    Args:
        imgs (np.ndarray): Matrix of samples x pixel values to whiten.
        return_zca_mat (bool): True to return the transformation matrix only, False
            to return the whitened images only (default).
        full_matrix (bool): Whether to use all of the SVD components (refer to
            https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
            for help).
    Returns:
        ZCAMatrix (np.ndarray) or whitened image matrix (np.ndarray) depending on return_zca_mat.
    '''

    if len(imgs.shape) != 2:
        raise ValueError('Arg imgs should be a matrix with 2 dims, but has shape \
                          {} instead.'.format(imgs.shape))

    imgs = imgs - np.mean(imgs, 0)
    U,S,V = np.linalg.svd(np.matmul(imgs, imgs.transpose()), full_matrices = full_matrix)
    epsilon = 1e-6
    ZCAMatrix = np.matmul(U, np.matmul(np.diag(1.0 / np.sqrt(S + epsilon)), U.transpose()))

    if return_zca_mat:
        return ZCAMatrix
    else:
        return np.matmul(ZCAMatrix, imgs)


def max_min_scale(array, eps = 1e-8):
    '''
    Linearly scale the values in an array to the range [0, 1].
    Args:
        array (np.ndarray): The n-dimensional array to scale.
        eps (float): A small value added to the denominator to avoid division by 0.
    Returns:
        np.ndarray with max value of 1.0 and min value of 0.0.
    '''

    if type(array) != np.ndarray:
        raise TypeError('Arg array should be of type np.ndarray but is of type \
                         {} instead.'.format(type(array)))

    return (array - np.amin(array)) / (np.amax(array) - np.amin(array) + eps)


def make_lgn_freq_filter(x_dim, y_dim, f_0 = 300):
    '''
    Frequency spectrum filter used in
    https://www.frontiersin.org/articles/10.3389/fncir.2019.00013/full

    Args:
        x_dim (int): Width of the images.
        y_dim (int): Height of the images.
        f_0 (int): Cutoff for cycles/s. Should be about 0.4 * min(x_dim, y_dim)
            for biologically-realistic acuity.

    Returns:
        A filter for the fft transformed image.
    '''
    y_vals = np.arange(-y_dim // 2, y_dim // 2)
    x_vals = np.arange(-x_dim // 2, x_dim // 2)
    freqs_x, freqs_y = np.meshgrid(x_vals, y_vals)
    freq_grid = np.sqrt(freqs_x ** 2 + freqs_y ** 2)
    freq_filter = freq_grid * np.exp(-(freq_grid / f_0) ** 4)

    return freq_filter
