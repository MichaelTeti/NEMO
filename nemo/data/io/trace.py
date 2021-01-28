import os

import pandas as pd


def load_single_cell_avg_traces(fpath, n_frames_in_time = 9):
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