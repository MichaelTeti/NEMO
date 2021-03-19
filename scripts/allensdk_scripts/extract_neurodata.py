from argparse import ArgumentParser
import logging
import os

import allensdk
from allensdk.brain_observatory.brain_observatory_exceptions import (
    EpochSeparationException,
    MissingStimulusException,
    NoEyeTrackingException
)
from allensdk.brain_observatory.locally_sparse_noise import LocallySparseNoise
import allensdk.brain_observatory.stimulus_info as si
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import cv2
import h5py
import multiprocessing as mp
import numpy as np
import pandas as pd
from progressbar import ProgressBar

from nemo.data.io.image import write_AIBO_natural_stimuli, write_AIBO_static_grating_stimuli
from nemo.data.preprocess.image import max_min_scale
from nemo.data.preprocess.trace import normalize_traces
from nemo.data.utils import get_img_frame_names, monitor_coord_to_image_ind
from nemo.utils import multiproc



def add_video_data_to_stim_table(df, data, stimulus):
    ''' Adds trace and behavioral data to natural movie stimulus table '''
    
    # get acquisition frame indices
    inds = list(df['end'])
    df = df.drop(columns = ['start', 'end'])

    # add data to stimulus table 
    df['stimulus'] = [stimulus] * len(inds)

    # needs to be compatible with image stimuli dtype
    df = df.astype({'repeat': np.float32})

    # make compatible with static gratings
    df['orientation'] = [np.nan] * len(inds)
    df['spatial_frequency'] = [np.nan] * len(inds)
    df['phase'] = [np.nan] * len(inds)

    for key, value in data.items():
        if key in ['cell_ids', 'dff']:
            continue 

        if type(value) == np.ndarray:
            df[key] = value[inds]
        else:
            df[key] = [value] * len(inds)

    # need to sort column names now to make sure they line up with image stimuli
    df.sort_index(axis = 1, inplace = True)
    
    for cell_traces, cell_id in zip(data['dff'].transpose(), data['cell_ids']):
        df['dff_{}'.format(cell_id)] = cell_traces[inds]
        

    return df


def add_image_data_to_stim_table(df, data, stimulus):
    ''' Adds trace and behavioral data to natural scene and image stimulus table '''
    
    list_add = []
    repeats = {}
    for row_num in range(len(df)):
        row = df.iloc[row_num].to_list()
        start, end = row[-2:]

        if stimulus == 'natural_scenes':
            frame = row[0]

        elif stimulus == 'static_gratings':
            orientation, spatial_freq, phase = row[:-2]
            frame = '{}_{}_{}'.format(orientation, spatial_freq, phase)
            
        if frame not in repeats.keys():
            repeats[frame] = 0.0
            
        repeat = repeats[frame]
        repeats[frame] += 1.0
        
        for ind in range(int(start), int(end)):
            list_add.append(row[:-2] + [ind, ind + 1, repeat])
            repeat += 0.1
            
    better_df = pd.DataFrame(list_add, columns = list(df.columns)[:-2] + ['start', 'end', 'repeat'])
    
    # get acquisition frame inds
    inds = list(better_df['end'])
    better_df = better_df.drop(columns = ['start', 'end'])

    # add data to stimulus table
    better_df['stimulus'] = [stimulus] * len(inds)

    # make natural scenes compatible with gratings and gratings compatible
    # with scenes and videos
    if stimulus == 'natural_scenes':
        better_df['orientation'] = [np.nan] * len(inds)
        better_df['spatial_frequency'] = [np.nan] * len(inds)
        better_df['phase'] = [np.nan] * len(inds)
    elif stimulus == 'static_gratings':
        better_df['frame'] = [np.nan] * len(inds)

    for key, value in data.items():
        if key in ['cell_ids', 'dff']:
            continue 

        if type(value) == np.ndarray:
            better_df[key] = value[inds]
        else:
            better_df[key] = [value] * len(inds)

    # need to sort column names now to make sure they line up with video stimuli
    better_df.sort_index(axis = 1, inplace = True)
    
    for cell_traces, cell_id in zip(data['dff'].transpose(), data['cell_ids']):
        better_df['dff_{}'.format(cell_id)] = cell_traces[inds]

    return better_df


def get_eye_tracking_data(dataset, missing_data_fill_size):
    ''' Gets the eye tracking data from the AIBO dataset. '''

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
    ''' Get the relevant data for an experiment. '''

    data = {}

    # get the cell IDs in the dataset 
    data['cell_ids'] = dataset.get_cell_specimen_ids()

    # get experiment ID and container ID
    data['cont_id'] = dataset.get_metadata()['experiment_container_id']

    # get session type 
    data['session_type'] = dataset.get_session_type()

    # get cre line
    data['cre_line'] = dataset.get_metadata()['cre_line'].split('/')[0]

    # get df/f  for all cells in this experiment
    # after these lines, traces is of shape # acquisition frames x # cells
    dff_ts, dff = dataset.get_dff_traces(cell_specimen_ids = data['cell_ids'])
    dff = np.float16(normalize_traces(dff.transpose()))
    data['dff'], data['ts'] = dff, dff_ts

    # get eye tracking info for the animal in this experiment
    data.update(
        get_eye_tracking_data(
            dataset,
            missing_data_fill_size = dff.shape[0]
        )
    ) 

    # get the running speed 
    data['run_speed'] = dataset.get_running_speed()[1]

    if data['run_speed'].shape[0] != dff.shape[0] or dff.shape[0] != data['pupil_x'].shape[0]:
        raise ValueError

    return data


def extract_exp_data(dataset, stimuli_dir):
    ''' Compiles experiment data across stimulus presentations in a dataframe. '''

    try:
        stim_epoch_table = dataset.get_stimulus_epoch_table()
    except EpochSeparationException:
        # this only happens a few cases, so not really worth it to try to fix this
        return
    

    # get the data
    aibo_data = get_AIBO_data(dataset)


    # combine data from each stimulus type
    stim_dfs = []
    for stimulus in dataset.list_stimuli():
        # stim_frame_table tells you how to index the trace, run, pupil, 
        # etc. data
        stim_frame_table = dataset.get_stimulus_table(stimulus)


        # writing traces, pupil coords, etc.
        if 'natural_movie' in stimulus:
            stim_dfs.append(
                add_video_data_to_stim_table(
                    df = stim_frame_table,
                    data = aibo_data,
                    stimulus = stimulus
                )
            )

        elif stimulus in ['natural_scenes', 'static_gratings']:
            stim_dfs.append(
                add_image_data_to_stim_table(
                    df = stim_frame_table,
                    data = aibo_data,
                    stimulus = stimulus
                )
            )
            

    return pd.concat(
        stim_dfs,
        axis = 0,
        join = 'outer',  # should not be necessary here, but shouldn't hurt
        ignore_index = True,
        verify_integrity = True
    )


def merge_datasets_in_cont(cont_of_datasets, save_dir):
    ''' Merges all dataframes for all experiments in a container '''

    dfs = []

    for dset_num, dataset in enumerate(cont_of_datasets):
        logging.info('EXTRACTING DATA FROM EXPERIMENT {}'.format(
            dataset.get_metadata()['ophys_experiment_id']
        ))
        dfs.append(
            extract_exp_data(
                dataset = dataset,
                stimuli_dir = os.path.join(save_dir, 'Stimuli')
            )    
        )

    return pd.concat(dfs, axis = 0, sort = True, join = 'outer')


def write_rfs(dataset, write_dir):
    '''
    https://allensdk.readthedocs.io/en/latest/_static/examples/nb/brain_observatory_analysis.html
    '''
    
    cell_ids = dataset.get_cell_specimen_ids()
    session_type = dataset.get_session_type()
    
    if 'locally_sparse_noise' in dataset.list_stimuli():
        logging.info('WRITING RECEPTIVE FIELDS FOR EXPERIMENT {}'.format(
            dataset.get_metadata()['ophys_experiment_id'])
        )

        on_dir = os.path.join(write_dir, 'ReceptiveFields', 'on', session_type)
        off_dir = os.path.join(write_dir, 'ReceptiveFields', 'off', session_type)
        os.makedirs(on_dir, exist_ok = True)
        os.makedirs(off_dir, exist_ok = True)

        lsn = LocallySparseNoise(dataset)
        rfs = lsn.receptive_field
        rfs[np.isnan(rfs)] = 0
        
        for ind, cell_id in enumerate(cell_ids):
            rf = rfs[:, :, ind, :]
            rf = max_min_scale(rf) * 255
            on, off = rf.transpose([2, 0, 1])
            fname = str(cell_id) + '.png'

            cv2.imwrite(os.path.join(on_dir, fname), on)
            cv2.imwrite(os.path.join(off_dir, fname), off)


def main(args):

    logging.basicConfig(
        format='%(levelname)s -- %(asctime)s -- %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p', 
        level = logging.INFO
    )
    
    boc = BrainObservatoryCache(manifest_file = args.manifest_fpath)
    exp_ids = [int(os.path.splitext(exp)[0]) for exp in os.listdir(args.exp_dir)]
    datasets = [boc.get_ophys_experiment_data(exp_id) for exp_id in exp_ids]


    if not args.no_trace_data:
        trace_dir = os.path.join(args.save_dir, 'NeuralData')
        os.makedirs(trace_dir, exist_ok = True)

        # keep datasets from same container together, bc this means that a single process will
        # loop through all datasets from a single container and combine them, which means we can use 
        # concat instead of merge, which sometimes takes very long for large dataframes
        cont_ids = list(set([dataset.get_metadata()['experiment_container_id'] for dataset in datasets]))
        datasets_by_cont = [
            [dataset for dataset in datasets if dataset.get_metadata()['experiment_container_id'] == cont_id] 
            for cont_id in cont_ids
        ]

        dfs = multiproc(
            func = merge_datasets_in_cont,
            iterator_keys = ['cont_of_datasets'],
            n_procs = args.n_workers,
            cont_of_datasets = datasets_by_cont,
            save_dir = args.save_dir
        )
        merge_cont_dfs_and_write(dfs, trace_dir)


    if not args.no_rfs:
        logging.info('WRITING RECEPTIVE FIELDS TO DISK')

        multiproc(
            func = write_rfs,
            iterator_keys = ['dataset'],
            n_procs = args.n_workers,
            dataset = datasets,
            write_dir = args.save_dir
        )

    
    if not args.no_stim:
        logging.info('WRITING STIMULI TEMPLATES TO DISK')

        stimuli_dir = os.path.join(args.save_dir, 'Stimuli')
        os.makedirs(stimuli_dir, exist_ok = True)

        # need to loop through datasets for static gratings,
        # bc some datasets have a few gratings not present 
        # in others sometimes
        multiproc(
            func = write_AIBO_static_grating_stimuli,
            iterator_keys = ['stim_table'],
            n_procs = args.n_workers,
            stim_table = [
                dset.get_stimulus_table('static_gratings') \
                for dset in datasets if 'static_gratings' in dset.list_stimuli()
            ],
            save_dir = os.path.join(stimuli_dir, 'static_gratings')
        )

        stimuli = [
            'natural_movie_one', 
            'natural_movie_two', 
            'natural_movie_three', 
            'natural_scenes'
        ]
        templates_by_stim = [
            [dset for dset in datasets if stim in dset.list_stimuli()][0].get_stimulus_template(stim) \
            for stim in stimuli
        ]
        multiproc(
            func = write_AIBO_natural_stimuli,
            iterator_keys = ['template', 'save_dir', 'stimulus'],
            n_procs = 4,
            template = templates_by_stim,
            save_dir = [os.path.join(stimuli_dir, stim) for stim in stimuli],
            stimulus = stimuli
        )
            



def merge_cont_dfs_and_write(dfs, trace_dir):
    ''' Merges dataframes from all containers and writes to disk '''

    logging.info('WRITING EXPERIMENT DATA TO DISK')

    stim_cols = [
        'frame', 
        'repeat', 
        'stimulus',  
        'session_type',
        'orientation', 
        'spatial_frequency', 
        'phase'
    ]
    loop_cols = [col for col in dfs[0].columns if (col not in stim_cols + ['cont_id'] and 'dff' not in col)]

    for col_name in loop_cols + ['dff']:
        for n_dfs, df in enumerate(dfs):
            cont_id = df.cont_id.to_list()[0]

            if col_name == 'dff':
                dff_cols = [col for col in df.columns if 'dff' in col]
                df_col = df[stim_cols + dff_cols]
                df_col = df_col.rename(
                    columns = dict(zip(dff_cols, [col[4:] + '_' + str(cont_id) for col in dff_cols]))
                )
            else:
                df_col = df[stim_cols + [col_name]]
                df_col = df_col.rename(columns = {col_name: '{}_{}'.format(col_name, cont_id)})

            if n_dfs == 0:
                df_agg = df_col 
            else:
                df_agg = pd.merge(
                    df_agg,
                    df_col,
                    how = 'outer',
                    copy = False
                )

        df_agg.to_hdf(
            os.path.join(trace_dir, col_name + '.h5'),
            key = 'data',
            complevel = 7
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
        '--no_trace_data',
        action = 'store_true',
        help = 'If specified, will not write out trace data.'
    )
    parser.add_argument(
        '--no_rfs',
        action = 'store_true',
        help = 'If specified, will not write out receptive fields.'
    )
    parser.add_argument(
        '--no_stim',
        action = 'store_true',
        help = 'If specified, will not write out stimuli templates.'
    )
    parser.add_argument(
        '--n_workers',
        type = int,
        default = 4,
        help = 'Number of datasets to write in parallel.'
    )
    args = parser.parse_args()

    main(args)