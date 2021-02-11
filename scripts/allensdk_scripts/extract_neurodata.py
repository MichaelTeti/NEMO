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
import cv2
import h5py
import numpy as np
import pandas as pd
from progressbar import ProgressBar

from nemo.data.io.image import write_AIBO_natural_stimuli, write_AIBO_static_grating_stimuli
from nemo.data.preprocess.trace import normalize_traces
from nemo.data.utils import get_img_frame_names, monitor_coord_to_image_ind



def write_natural_movie_data(write_dir, df, pupil_x, pupil_y, pupil_size, run_speed, dff, 
                             ts, cell_ids, session_type, stimulus):

    '''
    Writes response and behavioral data to file for natural movie stimuli.
    '''
    
    os.makedirs(write_dir, exist_ok = True)
    
    inds = list(df['end'])
    df['pupil_x'] = pupil_x[inds]
    df['pupil_y'] = pupil_y[inds]
    df['pupil_size'] = pupil_size[inds]
    df['run_speed'] = run_speed[inds]
    df['session_type'] = [session_type] * len(inds)
    df['stimulus'] = [stimulus] * len(inds)
    df['ts'] = ts[inds]
    
    for cell_traces, cell_id in zip(dff.transpose(), cell_ids):
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


def write_static_image_data(write_dir, df, pupil_x, pupil_y, pupil_size, run_speed, dff,
                            ts, cell_ids, session_type):

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
    better_df['pupil_x'] = pupil_x[inds]
    better_df['pupil_y'] = pupil_y[inds]
    better_df['pupil_size'] = pupil_size[inds]
    better_df['run_speed'] = run_speed[inds]
    better_df['ts'] = ts[inds]
    better_df['session_type'] = [session_type] * len(inds)
    
    for cell_traces, cell_id in zip(dff.transpose(), cell_ids):
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


def get_eye_tracking_info(dataset, missing_data_fill_size, keep_no_eye_tracking = False):
    '''
    Gets the eye tracking data from the AIBO dataset.
    '''

    try:
        pupil_ts, pupil_loc = dataset.get_pupil_location(as_spherical = False)
        _, pupil_size = dataset.get_pupil_size()
    except NoEyeTrackingException:
        if keep_no_eye_tracking:
            pupil_loc = np.full([missing_data_fill_size, 2], np.nan)
            pupil_size = np.full([missing_data_fill_size], np.nan)
            pupil_ts = np.full([missing_data_fill_size], np.nan)
        else:
            return

    pupil_x = pupil_loc[:, 0]
    pupil_y = pupil_loc[:, 1]
    pupil_x, pupil_y = monitor_coord_to_image_ind(pupil_x, pupil_y)

    return {
        'pupil_x': pupil_x,
        'pupil_y': pupil_y,
        'pupil_size': pupil_size,
        'pupil_ts': pupil_ts
    }


def extract_exp_data(dataset, trace_dir, stimuli_dir, keep_no_eye_tracking = False):

    try:
        stim_epoch_table = dataset.get_stimulus_epoch_table()
    except EpochSeparationException:
        # this only happens a few cases, so not really worth it to try to fix this
        return


    # get cell IDs in this experiment
    cell_ids = dataset.get_cell_specimen_ids()
        
        
    # get df/f  for all cells in this experiment
    trace_ts, traces = dataset.get_dff_traces(cell_specimen_ids = cell_ids)
    traces = traces.transpose()
    traces = normalize_traces(traces)


    # get eye tracking info for the animal in this experiment
    eye_data = get_eye_tracking_info(
        dataset,
        missing_data_fill_size = traces.shape[0], 
        keep_no_eye_tracking = keep_no_eye_tracking
    )
    if not eye_data: return


    # get running speed of animal in this experiment
    _, run_speed = dataset.get_running_speed()
    

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
                pupil_x = eye_data['pupil_x'],
                pupil_y = eye_data['pupil_y'],
                pupil_size = eye_data['pupil_size'],
                run_speed = run_speed,
                dff = traces,
                ts = trace_ts,
                cell_ids = cell_ids,
                session_type = dataset.get_session_type(),
                stimulus = stimulus
            )
        elif stimulus in ['natural_scenes', 'static_gratings']:
            write_static_image_data(
                write_dir = os.path.join(trace_dir, stimulus),
                df = stim_frame_table,
                pupil_x = eye_data['pupil_x'],
                pupil_y = eye_data['pupil_y'],
                pupil_size = eye_data['pupil_size'],
                run_speed = run_speed,
                dff = traces,
                ts = trace_ts,
                cell_ids = cell_ids,
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


def loop_exps(manifest_fpath, exp_dir, save_dir, keep_no_eye_tracking = False):
    boc = BrainObservatoryCache(manifest_file = manifest_fpath)
    exp_ids = [int(os.path.splitext(exp)[0]) for exp in os.listdir(exp_dir)]

    for exp_id in ProgressBar()(exp_ids):
        extract_exp_data(
            dataset = boc.get_ophys_experiment_data(exp_id),
            trace_dir = os.path.join(save_dir, 'trace_data'),
            stimuli_dir = os.path.join(save_dir, 'stimuli'),
            keep_no_eye_tracking = keep_no_eye_tracking
        )


def extract_rfs(manifest_fpath, exp_dir, write_dir):
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
                
                with h5py.File(write_fpath, 'a') as h5file:
                    if str(cell_id) not in list(h5file.keys()):
                        h5file.create_dataset(str(cell_id), data = rf)



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
        '--keep_no_eye_tracking',
        action = 'store_true',
        help = 'If specified, write data with no eye tracking available.'
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
            save_dir = args.save_dir,
            keep_no_eye_tracking = args.keep_no_eye_tracking
        )

    if not args.no_rfs:
        logging.info('Writing receptive fields')
        extract_rfs(
            manifest_fpath = args.manifest_fpath,
            exp_dir = args.exp_dir, 
            write_dir = args.save_dir
        )