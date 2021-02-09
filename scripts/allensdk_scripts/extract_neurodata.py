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

from nemo.data.preprocess.trace import normalize_traces
from nemo.data.utils import get_img_frame_names



def write_natural_video_stimuli(dataset, stimulus, save_dir, monitor):
    '''
    Obtains and writes the natural movie stimuli from the AIBO database.
    '''

    save_dir = os.path.join(save_dir, stimulus)
    os.makedirs(save_dir, exist_ok = True)
    
    template = dataset.get_stimulus_template(stimulus)
    n_frames = template.shape[0]
    fnames = [fname + '.png' for fname in get_img_frame_names(n_frames)]

    for fname, frame in zip(fnames, template):
        frame_screen = monitor.natural_movie_image_to_screen(frame, origin = 'upper')
        frame_warp = monitor.warp_image(frame_screen)
        cv2.imwrite(os.path.join(save_dir, fname), frame_warp)


def write_natural_image_stimuli(dataset, save_dir, monitor):
    '''
    Obtains and writes the natural scene stimuli from the AIBO database.
    '''

    save_dir = os.path.join(save_dir, 'natural_scenes')
    os.makedirs(save_dir, exist_ok = True)
    
    imgs = dataset.get_stimulus_template('natural_scenes')
    fnames = [fname + '.png' for fname in get_img_frame_names(imgs.shape[0])]
    
    for fname, img in zip(fnames, imgs):
        img_screen = monitor.natural_scene_image_to_screen(img, origin = 'upper')
        img_warp = monitor.warp_image(img_screen)
        cv2.imwrite(os.path.join(save_dir, fname), img_warp)


def write_static_grating_stimuli(dataset, stim_table, save_dir, monitor):
    '''
    Obtains and writes the static grating stimuli from the AIBO database.
    '''

    save_dir = os.path.join(save_dir, 'static_gratings')
    os.makedirs(save_dir, exist_ok = True)
    
    for orient in stim_table['orientation'].unique():
        for freq in stim_table['spatial_frequency'].unique():
            for phase in stim_table['phase'].unique():
                if np.isnan(orient) or np.isnan(freq) or np.isnan(phase):
                    continue
                    
                fname = '{}_{}_{}.png'.format(orient, freq, phase)
                if fname not in os.listdir(save_dir):
                    grating = monitor.grating_to_screen(
                        phase = phase, 
                        spatial_frequency = freq, 
                        orientation = orient
                    )
                    grating_warp = monitor.warp_image(grating)
                    cv2.imwrite(os.path.join(save_dir, fname), grating_warp)


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
        
        fname = '{}.txt'.format(cell_id)
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
    
    better_df = pd.DataFrame(columns = list(df.columns)[:-2] + ['start', 'end'])
    
    list_add = []
    for row_num in range(len(df)):
        row = df.iloc[row_num].to_list()
        start, end = row[-2:]
        
        for ind in range(int(start), int(end)):
            list_add.append(row[:-2] + [ind, ind + 1])
            
    better_df = better_df.append(pd.DataFrame(list_add, columns = better_df.columns))
    
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
        
        fname = '{}.txt'.format(cell_id)
        df_write.to_csv(
            os.path.join(write_dir, fname), 
            index = False, 
            mode = 'a',
            header = False if fname in os.listdir(write_dir) else True,
            na_rep = np.nan
        )


def monitor_cm_coords_to_image(x_cm, y_cm, monitor_w_cm = 51.0, monitor_h_cm = 32.5, 
                              img_w_pix = 1920, img_h_pix = 1200):

    '''
    Maps eye tracking coords in monitor dims to image dims.
    '''

    x_cm = (monitor_w_cm / 2) + x_cm
    y_cm = (monitor_h_cm / 2) + y_cm
    x_img = x_cm * img_w_pix / monitor_w_cm
    y_img = y_cm * img_h_pix / monitor_h_cm
    x_frac = x_img / img_w_pix
    y_frac = y_img / img_h_pix
    
    if np.amin(x_frac) < 0 or np.amax(x_frac) >= 1:
        raise ValueError
    if np.amin(y_frac) < 0 or np.amax(y_frac) >= 1:
        raise ValueError
    
    return x_frac, y_frac


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
    pupil_x, pupil_y = monitor_cm_coords_to_image(pupil_x, pupil_y)
    pupil_x, pupil_y = np.round(pupil_x, 4), np.round(pupil_y, 4)

    return pupil_x, pupil_y, pupil_size, pupil_ts


def extract_neurodata(manifest_fpath, exp_dir, save_dir, keep_no_eye_tracking = False):
    stimuli_dir = os.path.join(save_dir, 'stimuli')
    trace_dir = os.path.join(save_dir, 'trace_data')
    
    boc = BrainObservatoryCache(manifest_file = manifest_fpath)
    exps = os.listdir(exp_dir)
    exp_ids = [int(os.path.splitext(exp)[0]) for exp in exps]

    monitor = si.BrainObservatoryMonitor()
    
    for exp_id in ProgressBar()(exp_ids):
        dataset = boc.get_ophys_experiment_data(exp_id)
        cell_ids = dataset.get_cell_specimen_ids()
        
        try:
            stim_epoch_table = dataset.get_stimulus_epoch_table()
        except EpochSeparationException:
            continue
        
        # get running speed of animal
        run_ts, run_speed = dataset.get_running_speed()
        
        
        # get eye tracking info
        eye_data = get_eye_tracking_info(
            dataset,
            missing_data_fill_size = run_speed.shape[0], 
            keep_no_eye_tracking = keep_no_eye_tracking
        )
        if eye_data:
            pupil_x, pupil_y, pupil_size, pupil_ts = eye_data
        else:
            continue
            
            
        # get df/f corrected traces
        trace_ts, traces = dataset.get_dff_traces(cell_specimen_ids = cell_ids)
        traces = traces.transpose()
        traces = normalize_traces(traces)
        
                
        for stimulus in dataset.list_stimuli():
            stim_frame_table = dataset.get_stimulus_table(stimulus)
            
            # writing traces, pupil coords, etc.
            if 'natural_movie' in stimulus:
                write_natural_movie_data(
                    os.path.join(trace_dir, 'natural_movies'),
                    stim_frame_table,
                    pupil_x,
                    pupil_y,
                    pupil_size,
                    run_speed,
                    traces,
                    trace_ts,
                    cell_ids,
                    dataset.get_session_type(),
                    stimulus
                )
            elif stimulus in ['natural_scenes', 'static_gratings']:
                write_static_image_data(
                    os.path.join(trace_dir, stimulus),
                    stim_frame_table,
                    pupil_x,
                    pupil_y,
                    pupil_size,
                    run_speed,
                    traces,
                    trace_ts,
                    cell_ids,
                    dataset.get_session_type(),
                )
                
                
            # writing the stimuli
            if stimulus == 'static_gratings':
                write_static_grating_stimuli(
                    dataset,
                    stim_frame_table,
                    stimuli_dir,
                    monitor
                )
            elif stimulus == 'drifting_gratings':
                continue
                # TODO
                
            # these only show up once in an experiment
            # so if already there, then don't need to 
            # write again
            else:
                if stimulus not in stimuli_dir:
                    if 'natural_movie' in stimulus:
                        write_natural_video_stimuli(
                            dataset, 
                            stimulus, 
                            stimuli_dir, 
                            monitor
                        )
                    elif stimulus == 'natural_scenes':
                        write_natural_image_stimuli(
                            dataset,
                            stimuli_dir,
                            monitor
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
        'manifest_path',
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
        extract_neurodata(
            manifest_fpath = args.manifest_path,
            exp_dir = args.exp_dir,
            save_dir = args.save_dir,
            keep_no_eye_tracking = args.keep_no_eye_tracking
        )

    if not args.no_rfs:
        logging.info('Writing receptive fields')
        extract_rfs(
            manifest_fpath = args.manifest_path,
            exp_dir = args.exp_dir, 
            write_dir = args.save_dir
        )