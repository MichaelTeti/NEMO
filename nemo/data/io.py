import csv
import os

import cv2
import h5py
import numpy as np
import pandas as pd

from nemo.data.utils import get_img_frame_names


def write_csv(items, fpath, mode = 'w'):
    '''
    Write a list to a .csv file.

    Args:
        items (list): A list of values (e.g. strings, floats, ints, etc.) or lists.
        fpath (str): Desired fpath for the .csv file.
        mode (str): Write mode. See https://docs.python.org/3.6/library/functions.html#open
              for details.

    Returns:
        None
    '''

    with open(fpath, mode) as f:
        writer = csv.writer(f, delimiter = ',')
        for item in items:
            if type(item) != list: item = [item]
            writer.writerow(item)


def read_h5_as_array(fpath):
    '''
    Read a .h5 file and return the items as a dictionary of np.ndarrays.

    Args:
        fpath (str): The path to the .h5 file.

    Returns:
        data (dict): A dictionary of keys and their corresponding arrays.
    '''

    with h5py.File(fpath, 'r+') as h5file:
        data = {}
        for key in list(h5file.keys()):
            data[key] = h5file[key][()]

    return data


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


def load_trial_avg_trace_array(fpath, n_frames_in_time = 9):
    '''
    Loads trial-averaged traces as a 1D array for a model. 

    Args:
        fpath (str): The file path to the .txt file with the trial-averaged traces.
        n_frames_in_time (int): The number of consecutive frames in a single sample. 

    Returns:
        traces (np.ndarray): A 1D array with the trial-averaged traces over stimulus frames.
    '''

    if n_frames_in_time <= 0:
        raise ValueError('n_frames_in_time should be > 0.')
    
    traces = pd.read_csv(fpath)
    traces = traces.iloc[:, n_frames_in_time - 1:].to_numpy()[0]
    
    return traces


def read_csv(fpath, remove_header = False, remove_footer = False, mode = 'r',):
    '''
    Read a .csv file and return the items as a list.

    Args:
        fpath (str): The path to the .csv file.
        remove_header (bool): True to return lines 1-n.
        remove_footer (bool): True to return lines 0-(n-1).
        mode (str): Mode to open the file in.

    Returns:
        A list of items (e.g. strs, floats, ints, lists, etc.)
    '''

    with open(fpath, mode) as f:
        reader = csv.reader(f, delimiter = ',')
        data = []

        try:
            for row_num, row in enumerate(reader[:-1] if remove_footer else reader):
                if row_num == 0 and remove_header: continue
                data.append(row[0] if len(row) == 1 else row)
        except:
            pass

    return data
    

def compile_trial_avg_traces(trace_dir):
    '''
    Reads in trial-averaged traces and aggregates them into a single dataframe.

    Args:
        trace_dir (str): The directory containing all of the cellID_*.txt trace files.

    Returns:
        df (pd.DataFrame): The dataframe of shape # neurons x # frames.
        cell_ids (list): A list of the cell IDs for those in the dataframe. 
    '''

    # get paths to all the trace files in the given trace_dir
    fnames = os.listdir(trace_dir)
    fnames.sort() # sort them so they'll be lined up regardless of stimuli etc.
    fpaths = [os.path.join(trace_dir, f) for f in fnames]
    cell_ids = [os.path.splitext(fname)[0].split('_')[1] for fname in fnames]

    # loop through and read them all into a pandas dataframe
    for fpath_num, fpath in enumerate(fpaths):
        if fpath_num == 0:
            df = pd.read_csv(fpath)
        else:
            df = df.append(pd.read_csv(fpath))

    # reset the indices 
    df = df.reset_index(drop = True)

    return df, cell_ids
