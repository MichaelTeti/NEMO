from argparse import ArgumentParser
import logging
import os

import allensdk
from allensdk.brain_observatory.brain_observatory_exceptions import (
    EpochSeparationException,
    NoEyeTrackingException
)
from allensdk.brain_observatory.locally_sparse_noise import LocallySparseNoise
import allensdk.brain_observatory.stimulus_info as si
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import h5py
import numpy as np
import pandas as pd
from progressbar import ProgressBar

from nemo.data.io.image import write_AIBO_natural_stimuli, write_AIBO_static_grating_stimuli
from nemo.data.preprocess.image import max_min_scale
from nemo.data.preprocess.trace import normalize_traces
from nemo.data.utils import get_img_frame_names, monitor_coord_to_image_ind



def write_natural_movie_data(write_dir, df, data, session_type, stimulus):

    '''
    Writes response and behavioral data to file for natural movie stimuli.
    '''
    
    os.makedirs(write_dir, exist_ok = True)
    
    inds = list(df['end'])
    df['pupil_x'] = data['pupil_x'][inds]
    df['pupil_y'] = data['pupil_y'][inds]
    df['pupil_size'] = data['pupil_size'][inds]
    df['run_speed'] = data['run_speed'][inds]
    df['session_type'] = [session_type] * len(inds)
    df['stimulus'] = [stimulus] * len(inds)
    df['dff_ts'] = data['dff_ts'][inds]
    
    for cell_traces, cell_id in zip(data['dff'].transpose(), data['cell_ids']):
        df_write = df.copy()
        df_write['dff'] = cell_traces[inds]
        
        fname = '{}.xz'.format(cell_id)
        df_write.to_csv(
            os.path.join(write_dir, fname), 
            index = False, 
            mode = 'a',
            header = False if fname in os.listdir(write_dir) else True,
            na_rep = np.nan
        )


def write_static_image_data(write_dir, df, data, session_type):

    '''
    Writes response and behavioral data to file for static image data.
    '''
    
    os.makedirs(write_dir, exist_ok = True)
    
    list_add = []
    for row_num in range(len(df)):
        row = df.iloc[row_num].to_list()
        start, end = row[-2:]
        
        for ind in range(int(start), int(end)):
            list_add.append(row[:-2] + [ind, ind + 1])
            
    better_df = pd.DataFrame(list_add, columns = list(df.columns)[:-2] + ['start', 'end'])
    
    inds = list(better_df['end'])
    better_df['pupil_x'] = data['pupil_x'][inds]
    better_df['pupil_y'] = data['pupil_y'][inds]
    better_df['pupil_size'] = data['pupil_size'][inds]
    better_df['run_speed'] = data['run_speed'][inds]
    better_df['dff_ts'] = data['dff_ts'][inds]
    better_df['session_type'] = [session_type] * len(inds)
    
    for cell_traces, cell_id in zip(data['dff'].transpose(), data['cell_ids']):
        df_write = better_df.copy()
        df_write['dff'] = cell_traces[inds]
        
        fname = '{}.xz'.format(cell_id)
        df_write.to_csv(
            os.path.join(write_dir, fname), 
            index = False, 
            mode = 'a',
            header = False if fname in os.listdir(write_dir) else True,
            na_rep = np.nan
        )


def get_eye_tracking_data(dataset, missing_data_fill_size):
    '''
    Gets the eye tracking data from the AIBO dataset.
    '''

    try:
        _, pupil_loc = dataset.get_pupil_location(as_spherical = False)
        _, pupil_size = dataset.get_pupil_size()

    except NoEyeTrackingException:
        pupil_loc = np.full([missing_data_fill_size, 2], np.nan)
        pupil_size = np.full([missing_data_fill_size], np.nan)

    finally:
        pupil_x = pupil_loc[:, 0]
        pupil_y = pupil_loc[:, 1]
        pupil_x, pupil_y = monitor_coord_to_image_ind(pupil_x, pupil_y)

    return {
        'pupil_x': pupil_x,
        'pupil_y': pupil_y,
        'pupil_size': pupil_size
    }


def get_AIBO_data(dataset):
    data = {}

    # get the cell IDs in the dataset 
    data['cell_ids'] = dataset.get_cell_specimen_ids()

    # get df/f  for all cells in this experiment
    # after these lines, traces is of shape # acquisition frames x # cells
    dff_ts, dff = dataset.get_dff_traces(cell_specimen_ids = data['cell_ids'])
    dff = normalize_traces(dff.transpose())
    data['dff'], data['dff_ts'] = dff, dff_ts

    # get eye tracking info for the animal in this experiment
    eye_data = get_eye_tracking_data(
        dataset, 
        missing_data_fill_size = dff.shape[0]
    )
    data.update(eye_data) 

    # get the running speed 
    data['run_speed'] = dataset.get_running_speed()[1]

    if data['run_speed'].shape[0] != dff.shape[0] or dff.shape[0] != eye_data['pupil_x'].shape[0]:
        raise ValueError

    return data


def write_exp_data(dataset, trace_dir, stimuli_dir):

    os.makedirs(stimuli_dir, exist_ok = True)
    os.makedirs(trace_dir, exist_ok = True)


    try:
        stim_epoch_table = dataset.get_stimulus_epoch_table()
    except EpochSeparationException:
        # this only happens a few cases, so not really worth it to try to fix this
        return
    

    # get the data
    aibo_data = get_AIBO_data(dataset)


    # write out data by stimulus
    for stimulus in dataset.list_stimuli():

        # stim_frame_table tells you how to index the trace, run, pupil, 
        # etc. data
        stim_frame_table = dataset.get_stimulus_table(stimulus)


        # writing traces, pupil coords, etc.
        if 'natural_movie' in stimulus:

            write_natural_movie_data(
                write_dir = os.path.join(trace_dir, 'natural_movies'),
                df = stim_frame_table,
                data = aibo_data,
                session_type = dataset.get_session_type(),
                stimulus = stimulus
            )

        elif stimulus in ['natural_scenes', 'static_gratings']:

            write_static_image_data(
                write_dir = os.path.join(trace_dir, stimulus),
                df = stim_frame_table,
                data = aibo_data,
                session_type = dataset.get_session_type(),
            )
            
            
        # writing the stimuli
        if stimulus == 'static_gratings':

            write_AIBO_static_grating_stimuli(
                stim_table = stim_frame_table,
                save_dir = os.path.join(stimuli_dir, stimulus)
            )

        elif 'natural' in stimulus:

            if stimulus not in os.listdir(stimuli_dir):
                write_AIBO_natural_stimuli(
                    template = dataset.get_stimulus_template(stimulus),
                    save_dir = os.path.join(stimuli_dir, stimulus),
                    stimulus = stimulus
                )


def loop_exps(manifest_fpath, exp_dir, save_dir):
    boc = BrainObservatoryCache(manifest_file = manifest_fpath)
    exp_ids = [int(os.path.splitext(exp)[0]) for exp in os.listdir(exp_dir)]

    for exp_id in ProgressBar()(exp_ids):
        write_exp_data(
            dataset = boc.get_ophys_experiment_data(exp_id),
            trace_dir = os.path.join(save_dir, 'trace_data'),
            stimuli_dir = os.path.join(save_dir, 'stimuli')
        )


def write_rfs(manifest_fpath, exp_dir, write_dir):
    '''
    https://allensdk.readthedocs.io/en/latest/_static/examples/nb/brain_observatory_analysis.html
    '''

    write_dir = os.path.join(write_dir, 'receptive_fields')
    write_fpath = os.path.join(write_dir, 'lsn_rfs.h5')
    os.makedirs(write_dir, exist_ok = True)
    
    boc = BrainObservatoryCache(manifest_file = manifest_fpath)
    exps = os.listdir(exp_dir)
    exp_ids = [int(os.path.splitext(exp)[0]) for exp in exps]
    
    for exp_id in ProgressBar()(exp_ids):
        dataset = boc.get_ophys_experiment_data(exp_id)
        cell_ids = dataset.get_cell_specimen_ids()
        
        if 'locally_sparse_noise' in dataset.list_stimuli():
            lsn = LocallySparseNoise(dataset)
            rfs = lsn.receptive_field
            rfs[np.isnan(rfs)] = 0
            
            for ind, cell_id in enumerate(cell_ids):
                rf = rfs[:, :, ind, :]
                on, off = rf.transpose([2, 0, 1])

                on = max_min_scale(on) * 255
                off = max_min_scale(off) * 255
                
                with h5py.File(write_fpath, 'a') as h5file:
                    if str(cell_id) not in list(h5file.keys()):
                        h5file.create_dataset(
                            str(cell_id), 
                            data = np.concatenate((on[None, ...], off[None, ...]), 0)
                        )



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'exp_dir',
        type = str,
        help = 'Directory containing the .nwb experiment files.'
    )
    parser.add_argument(
        'manifest_fpath',
        type = str,
        help = 'Path to the manifest.json file.'
    )
    parser.add_argument(
        'save_dir',
        type = str,
        help = 'Where to save all extracted data.'
    )
    parser.add_argument(
        '--no_stim_or_trace_data',
        action = 'store_true',
        help = 'If specified, will not write out stimuli or trace data.'
    )
    parser.add_argument(
        '--no_rfs',
        action = 'store_true',
        help = 'If specified, will not write out receptive fields.'
    )
    args = parser.parse_args()


    logging.basicConfig(
        format='%(asctime)s -- %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p', 
        level = logging.INFO
    )


    if not args.no_stim_or_trace_data:
        logging.info('Writing stimuli, traces, and behavioral data')
        loop_exps(
            manifest_fpath = args.manifest_fpath,
            exp_dir = args.exp_dir,
            save_dir = args.save_dir
        )

    if not args.no_rfs:
        logging.info('Writing receptive fields')
        write_rfs(
            manifest_fpath = args.manifest_fpath,
            exp_dir = args.exp_dir, 
            write_dir = args.save_dir
        )