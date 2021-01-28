import os

import cv2
import numpy as np

from nemo.data.utils import get_img_frame_names


def save_vid_array_as_frames(vid_arrays_and_save_dirs):
    '''
    Save a video represented as an array as individual frames.

    Args:
        vid_array_and_save_dir (list): A list of 2-tuples / lists, where each
            tuple is composed of a NxHxWxC (or NxHxW for grayscale) array and a
            dir to save the array at.

    Returns:
        None
    '''

    for vid_array, save_dir in vid_arrays_and_save_dirs:
        os.makedirs(save_dir, exist_ok = True)
        n_frames = vid_array.shape[0]
        fnames = [fname + '.png' for fname in get_img_frame_names(n_frames)]
        fpaths = [os.path.join(save_dir, fname) for fname in fnames]
        
        for i_frame, (frame, fpath) in enumerate(zip(vid_array, fpaths)):
            cv2.imwrite(fpath, np.uint8(frame))


def read_frames(dir, return_type = 'array', gray = False):
    '''
    Traverse a directory structure, reading in all images along the way and returning as list or np.ndarray.

    Args:
        dir (str): Directory to start at.
        return_type ('array' or 'list'): Data structure to return the video frames as.
        gray (bool): If true, return grayscale frames.

    Returns:
        Video frames.
    '''

    frames = []
    for root, dirs, files in os.walk(dir):
        files.sort()

        for file in files:

            if os.path.splitext(file)[1] in ['.jpeg', '.jpg', '.JPG', '.JPEG', '.PNG', '.png']:
                frame = cv2.imread(os.path.join(root, file))
                
                if gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
                frames.append(frame)

    if return_type == 'array':
        return np.array(frames)
    elif return_type == 'list':
        return frames