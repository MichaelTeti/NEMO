from argparse import ArgumentParser
import os, sys

from allensdk.brain_observatory.receptive_field_analysis.receptive_field import \
    compute_receptive_field_with_postprocessing as get_rf
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from cv2 import imwrite
import numpy as np
import pandas as pd

from NEMO.utils.image_utils import (
    max_min_scale,
    save_vid_array_as_frames,
    get_img_frame_names
)
from NEMO.utils.general_utils import (
    write_csv,
    multiproc,
    write_h5
)


def save_natural_video_stimuli(cell_data_list, save_dir, n_workers, stimuli):
    print('[INFO] SAVING STIMULI TEMPLATES NOW.')

    # Save these in a subdir called Stimuli
    save_dir = os.path.join(save_dir, 'Stimuli')
    os.makedirs(save_dir, exist_ok = True)

    vids_and_fpaths = []
    saved_stimuli = []

    for stimulus in stimuli:
        for dataset, _, _, _, _ in cell_data_list:
            if stimulus in saved_stimuli: continue

            try:
                vid = dataset.get_stimulus_template(stimulus)  # get stimulus template from dataset
            except KeyError:
                continue

            save_dir_stimulus = os.path.join(save_dir, stimulus)
            os.makedirs(save_dir_stimulus, exist_ok = True)
            vid = np.uint8(max_min_scale(vid) * 255) # scale the video frames to the range [0, 255]
            vids_and_fpaths.append([vid, save_dir_stimulus])
            saved_stimuli.append(stimulus)

    if vids_and_fpaths != []:
        multiproc(
            func = save_vid_array_as_frames,
            iterator_key = 'vid_arrays_and_save_dirs',
            n_workers = n_workers,
            vid_arrays_and_save_dirs = vids_and_fpaths
        )


def save_natural_video_traces(cell_data_list, save_dir, missing_pupil_coords_thresh, stimuli):
    print('[INFO] SAVING TRACES AND PUPIL COORDS NOW.')

    if missing_pupil_coords_thresh:
        assert((missing_pupil_coords_thresh > 0 and missing_pupil_coords_thresh < 1)), \
            'missing_pupil_coords_thresh should be > 0 and < 1.'

    # create dirs to save traces and pupil coords
    save_dir_traces = os.path.join(save_dir, 'Traces')
    save_dir_pupil = os.path.join(save_dir, 'PupilLocs')
    save_dir_run = os.path.join(save_dir, 'RunningSpeed')
    os.makedirs(save_dir_traces, exist_ok = True)
    os.makedirs(save_dir_pupil, exist_ok = True)
    os.makedirs(save_dir_run, exist_ok = True)

    for i_stimulus, stimulus in enumerate(stimuli):
        for dataset, container, experiment, cell_id, cell_ind in cell_data_list:
            if stimulus in dataset.list_stimuli():
                # get the stimulus presentation table for that stimulus
                stim_table = dataset.get_stimulus_table(stimulus_name = stimulus)

                # get the session type
                session_type = dataset.get_session_type()

                # get number of times the stimulus was repeated and number of frames
                n_trials = stim_table['repeat'].max() + 1
                n_frames = stim_table.iloc[-1]['frame'] + 1

                # get the corrected fluorescence traces for the current cell
                # and scale them over all stimuli
                trace_ts, traces = dataset.get_corrected_fluorescence_traces(cell_specimen_ids = [cell_id])
                traces = max_min_scale(traces, eps = 1e-12)
                traces -= np.mean(traces)

                # get pupil coordinates, but put nan if they aren't available
                try:
                    pupil_loc_ts, pupil_locs = dataset.get_pupil_location()
                except:
                    pupil_loc_ts = trace_ts
                    pupil_locs = np.zeros([trace_ts.size, 2])
                    pupil_locs[:, :] = np.nan

                # get running speed
                run_speed_ts, run_speed = dataset.get_running_speed()

                # create lists to append data to
                traces_agg = []
                pupil_agg = []
                run_agg = []

                # create a header for this stimulus to make dataframe later
                header = [
                    'container',
                    'experiment',
                    'cell_id',
                    'cell_ind',
                    'trial',
                    'stimulus',
                    'session_type'
                ]
                header += [frame_name + '.png' for frame_name in get_img_frame_names(n_frames)]

                # loop through each trial
                for i_trial in range(n_trials):
                    # find out where to index the traces and pupil locations
                    stim_table_trial = stim_table[stim_table['repeat'] == i_trial]
                    start_ind = stim_table_trial.iloc[0]['end']
                    end_ind = start_ind + n_frames

                    # index the pupil locations, traces, and running speed based on the trial
                    pupil_locs_trial = pupil_locs[start_ind:end_ind, :]
                    traces_trial = traces[0, start_ind:end_ind]
                    run_speed_trial = run_speed[start_ind:end_ind]

                    # filter out data with missing eye locations at a proportion greater than missing_pupil_coords_thresh
                    missing_prop_pupil = np.mean(np.isnan(pupil_locs_trial[:, 0]))
                    if missing_pupil_coords_thresh and (missing_prop_pupil > missing_pupil_coords_thresh):
                        traces_trial[:] = np.nan
                        pupil_locs_trial[:, :] = np.nan

                    # change the pupil locations to fractions between 0 and 1 so resizing the image won't change it
                    pupil_locs_trial[:, 0] = (pupil_locs_trial[:, 0] + 608 // 2) / 608
                    pupil_locs_trial[:, 1] = (pupil_locs_trial[:, 1] + 304 // 2) / 304

                    # add cell metadata to things to write
                    cell_metadata = [container, experiment, cell_id, cell_ind, i_trial, stimulus, session_type]
                    traces_cell = cell_metadata.copy()
                    pupil_coords_cell = cell_metadata.copy()
                    run_cell = cell_metadata.copy()

                    # add the data from this trial to the data for that cell
                    traces_cell += [round(float(trace), 2) for trace in traces_trial]
                    pupil_coords_cell += ['{}/{}'.format(round(float(x), 2), round(float(y), 2)) for x,y in pupil_locs_trial]
                    run_cell += [run_speed for run_speed in run_speed_trial]

                    # add to the lists for this stimuli and experiment
                    traces_agg.append(traces_cell)
                    pupil_agg.append(pupil_coords_cell)
                    run_agg.append(run_cell)

                # create pandas dataframes
                df_traces = pd.DataFrame(traces_agg, columns = header)
                df_pupil = pd.DataFrame(pupil_agg, columns = header)
                df_run = pd.DataFrame(run_agg, columns = header)

                # figure out where to save the responses
                save_fpath_traces = os.path.join(
                    save_dir_traces,
                    '{}'.format(stimulus),
                    '{}'.format(session_type)
                )
                save_fpath_pupil = os.path.join(
                    save_dir_pupil,
                    '{}'.format(stimulus),
                    '{}'.format(session_type)
                )
                save_fpath_run = os.path.join(
                    save_dir_run,
                    '{}'.format(stimulus),
                    '{}'.format(session_type)
                )
                os.makedirs(save_fpath_traces, exist_ok = True)
                os.makedirs(save_fpath_pupil, exist_ok = True)
                os.makedirs(save_fpath_run, exist_ok = True)

                # write out data
                df_traces.to_csv(
                    path_or_buf = os.path.join(save_fpath_traces, 'cellID_{}.txt'.format(cell_id)),
                    mode = 'w',
                    header = True,
                    index = False
                )
                df_pupil.to_csv(
                    path_or_buf = os.path.join(save_fpath_pupil, 'cellID_{}.txt'.format(cell_id)),
                    mode = 'w',
                    header = True,
                    index = False
                )
                df_run.to_csv(
                    path_or_buf = os.path.join(save_fpath_run, 'cellID_{}.txt'.format(cell_id)),
                    mode = 'w',
                    header = True,
                    index = False
                )



def save_receptive_fields(cell_data_list, alpha, save_dir, sig_chi_only):
    print('[INFO] SAVING RECEPTIVE FIELDS.')

    save_dir = os.path.join(save_dir, 'ReceptiveFieldImages')
    os.makedirs(save_dir, exist_ok = True)

    for i_cell, (dataset, container, experiment, cell_id, cell_ind) in enumerate(cell_data_list):
        if 'locally_sparse_noise' in dataset.list_stimuli():
            stimulus = 'locally_sparse_noise'
        elif 'locally_sparse_noise_4deg' in dataset.list_stimuli():
            stimulus = 'locally_sparse_noise_4deg'
        else:
            continue

        rf_data = get_rf(dataset, cell_ind, stimulus, alpha = alpha, number_of_shuffles = 10000)

        # only save significant ones if sig_chi_only is specified
        if sig_chi_only and not rf_data['chi_squared_analysis']['attrs']['significant']: continue
        chi_squared = 1.0 - rf_data['chi_squared_analysis']['pvalues']['data']
        chi_squared = chi_squared * np.float32(chi_squared  > 1.0 - alpha)
        on, off = 1.0 - rf_data['on']['pvalues']['data'], 1.0 - rf_data['off']['pvalues']['data']
        on, off = on * np.float32(on > 1.0 - alpha), off * np.float32(off > 1.0 - alpha)

        imwrite(os.path.join(save_dir, '{}_on.png'.format(cell_id)), np.uint8(max_min_scale(on) * 255))
        imwrite(os.path.join(save_dir, '{}_off.png'.format(cell_id)), np.uint8(max_min_scale(off) * 255))
        imwrite(os.path.join(save_dir, '{}_chi_squared.png'.format(cell_id)), np.uint8(max_min_scale(chi_squared) * 255))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'manifest_path',
        type = str,
        help = 'Path to the manifest file.'
    )
    parser.add_argument(
        'experiment_dir',
        type = str,
        help = 'Path to the folder called ophys_experiment_data.'
    )
    parser.add_argument(
        'save_dir',
        type = str,
        help = 'Directory to put the extracted data.'
    )
    parser.add_argument(
        '--stimuli',
        type = str,
        nargs = '+',
        choices = [
            'natural_movie_one',
            'natural_movie_two',
            'natural_movie_three',
        ],
        help = 'Stimuli to save templates and responses for if save_stimuli specified.'
    )
    parser.add_argument(
        '--n_workers',
        type = int,
        default = 4,
        help = 'Number of workers to use. Default = 4'
    )
    parser.add_argument(
        '--save_stimuli',
        action = 'store_true',
        help = 'If specified, will save stimuli.'
    )
    parser.add_argument(
        '--save_traces_and_pupil_coords',
        action = 'store_true',
        help = 'If specified, will save fluorescence traces and eye tracking data.'
    )
    parser.add_argument(
        '--save_rfs',
        action = 'store_true',
        help = 'If specified, will save RFs.'
    )
    parser.add_argument('--missing_pupil_coords_thresh',
        type = float,
        help = 'Will not save data files if the fraction of missing pupil coords is \
            above this value. Should be in the range (0, 1).')
    parser.add_argument('--rf_alpha',
        type = float,
        default = 0.05,
        help = 'The alpha level to use in the receptive field estimation. Default is 0.05.')
    parser.add_argument('--sig_chi_only',
        action = 'store_true',
        help = 'If specified, only the receptive fields with significant chi squared results will be saved.')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok = True)

    exp_ids = [int(f.split('.')[0]) for f in os.listdir(args.experiment_dir)]
    boc = BrainObservatoryCache(manifest_file = args.manifest_path)

    # get experiment data
    datasets = [boc.get_ophys_experiment_data(exp_id) for exp_id in exp_ids]

    # get cells in the experiment, along with the container id and experiment id
    data = []
    for dataset in datasets:
        cont = dataset.get_metadata()['experiment_container_id']
        exp = dataset.get_metadata()['ophys_experiment_id']
        cell_ids = list(dataset.get_cell_specimen_ids())
        cell_inds = dataset.get_cell_specimen_indices(cell_ids)
        data += [[dataset, cont, exp, cell_id, cell_ind] for cell_id, cell_ind in zip(cell_ids, cell_inds)]

    if args.save_stimuli:
        save_natural_video_stimuli(
            cell_data_list = data,
            save_dir = args.save_dir,
            n_workers = args.n_workers,
            stimuli = args.stimuli
        )

    if args.save_traces_and_pupil_coords:
        save_natural_video_traces(
            cell_data_list = data,
            save_dir = args.save_dir,
            missing_pupil_coords_thresh = args.missing_pupil_coords_thresh,
            stimuli = args.stimuli
        )

    if args.save_rfs:
        multiproc(
            func = save_receptive_fields,
            iterator_key = 'cell_data_list',
            n_workers = args.n_workers,
            cell_data_list = data,
            alpha = args.rf_alpha,
            save_dir = args.save_dir,
            sig_chi_only = args.sig_chi_only
        )


# python3 extract_neurodata.py \
#    ../../../BrainObservatoryData/manifest.json \
#    ../../../BrainObservatoryData/ophys_experiment_data/ \
#    ../../../BrainObservatoryData/ExtractedData \
#    --stimuli natural_movie_one natural_movie_three \
#    --n_workers 16 \
#    --save_traces_and_pupil_coords \
#    --save_stimuli \
#    --save_rfs
