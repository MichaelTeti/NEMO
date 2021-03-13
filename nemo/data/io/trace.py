from glob import glob
import os

import pandas as pd



def read_neural_data_file(fpath, stimuli = None):
    '''
    Reads in a neural data file given an fpath and returns the dataframe.

    Args:
        fpath (str): The path to the cell's data file to read in.
        stimuli (list): List of stimulus names to keep (e.g. natural_movie_one,
            static_gratings, etc.)

    Returns:
        df (pd.DataFrame): The cell's dataframe. If stimuli given, df.stimulus.unique()
            should be equal to stimuli.
        cell_id (str): The cell's ID in the AIBO database. 
    '''

    # read the file and get the cell name from the filename
    df = pd.read_csv(fpath)
    cell_id = os.path.splitext(os.path.split(fpath)[1])[0]

    # pull out desired stimuli
    if stimuli:
        if not all([stimulus in df.stimulus.unique() for stimulus in stimuli]):
            return pd.DataFrame(), cell_id
        else:
            df = df[df.stimulus.isin(stimuli)]

    return df, cell_id