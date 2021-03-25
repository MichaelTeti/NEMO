from functools import reduce

import numpy as np
import pandas as pd


def normalize_traces(traces):
    '''
    Zero-mean and scale traces.
    
    Args:
        traces (np.ndarray): The array of unscaled fluorescence trace values.
        
    Returns:
        traces_scaled (np.ndarray): The array of fluorescence traces with range [-1, 1].
    '''
    
    traces /= np.amax(np.absolute(traces), 0)
    
    return traces


def compute_trial_avgs(df):
    '''
    Computes average dff over repeated presentations.

    Args:
        df (pd.DataFrame): Dataframe with at least stimulus, frame, and dff
            columns.
    
    Returns:
        trial_avg_df (pd.DataFrame): Dataframe where dff is now averaged over
            repeated presentations of the same frame.
    '''

    try:
        df = df.groupby(['stimulus', 'frame']).mean().reset_index()
    except KeyError:
        print('df must have columns named stimulus and frame.')
        raise 
    else:
        df = df.sort_values(by = ['stimulus', 'frame']).reset_index(drop = True)

    return df


def aggregate_cell_data(dfs, cell_ids = None, keep_cols = None):
    '''
    Concatenates dfs.

    Args:
        dfs (list): List of pd.DataFrame, which all should have the same columns.
        cell_ids (list): Cell IDs corresponding to dfs. This will be used to rename
            each one's columns when concatenating dfs because we can't have the same
            column names.
        keep_cols (list): If a column name is in keep_cols, the column will kept
            separate after the merge, whereas columns not in keep_cols will be 
            merged into one column across dataframes.

    Returns:
        merged_df (pd.DataFrame): Merged dataframe.
    '''

    if cell_ids:
        if len(dfs) != len(cell_ids):
            raise ValueError('dfs is of different length than cell_ids')
    else:
        cell_ids = [str(num) for num in range(len(dfs))]


    # also need to rename each column we are interested in bc we can't have the same name
    if keep_cols is not None:
        dfs = [
            df.rename(columns = dict(zip(keep_cols, [cell_id + '_' + col for col in keep_cols]))) 
            for df, cell_id in zip(dfs, cell_ids)
        ]

    return reduce(lambda left, right: pd.merge(left, right, how = 'outer'), dfs)