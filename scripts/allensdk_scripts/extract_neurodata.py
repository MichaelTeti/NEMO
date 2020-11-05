from argparse import ArgumentParser
import os, sys

from allensdk.brain_observatory.receptive_field_analysis.receptive_field import \
    compute_receptive_field_with_postprocessing as get_rf
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from cv2 import imwrite
import numpy as np

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


def save_stimuli(cell_data_list, save_dir, n_workers, stimuli):
    print('[INFO] SAVING STIMULI TEMPLATES NOW.')

    # Save these in a subdir called Stimuli
    save_dir = os.path.join(save_dir, 'Stimuli')
    os.makedirs(save_dir, exist_ok = True)

    vids_and_fpaths = []

    for stimulus in stimuli:
        for dataset, _, _, _, _ in cell_data_list:
            if stimulus not in dataset.list_stimuli(): continue
            if stimulus not in os.listdir(save_dir):
                save_dir_stimulus = os.path.join(save_dir, stimulus)
                os.makedirs(save_dir_stimulus, exist_ok = True)
                vid = dataset.get_stimulus_template(stimulus)  # get stimulus template from dataset
                vid = np.uint8(max_min_scale(vid) * 255) # scale the video frames to the range [0, 255]
                vids_and_fpaths.append([vid, save_dir_stimulus])

    if vids_and_fpaths != []:
        multiproc(save_vid_array_as_frames, [vids_and_fpaths], n_workers = n_workers)


def save_traces_and_pupil_data(cell_data_list, save_dir, missing_pupil_coords_thresh, stimuli):
    print('[INFO] SAVING TRACES AND PUPIL COORDS NOW.')

    if missing_pupil_coords_thresh:
        assert((missing_pupil_coords_thresh > 0 and missing_pupil_coords_thresh < 1)), \
            'missing_pupil_coords_thresh should be > 0 and < 1.'

    # number of repeated trials is 10 in these experiments.
    n_trials = 10

    for i_stimulus, stimulus in enumerate(stimuli):
        # save the traces and pupil coords in respective directories
        save_fpath_stimulus_traces = os.path.join(save_dir, 'Traces', '{}.txt'.format(stimulus))
        save_fpath_stimulus_pupil = os.path.join(save_dir, 'PupilLocs', '{}.txt'.format(stimulus))

        # make sure those directories exist, create them if not
        os.makedirs(os.path.split(save_fpath_stimulus_traces)[0], exist_ok = True)
        os.makedirs(os.path.split(save_fpath_stimulus_pupil)[0], exist_ok = True)

        # keep a counter for the cells
        cell_count = 0

        for dataset, container, experiment, cell_id, cell_ind in cell_data_list:
            if stimulus in dataset.list_stimuli():
                # make empty lists to put the cell data in
                traces_cell, pupil_coords_cell = [], []

                # get the stimulus presentation table for that stimulus
                stim_table = dataset.get_stimulus_table(stimulus_name = stimulus)

                # get the corrected fluorescence traces
                trace_ts, traces = dataset.get_corrected_fluorescence_traces(cell_specimen_ids = [cell_id])

                # get pupil coordinates, but skip if they don't exist
                try:
                    pupil_loc_ts, pupil_locs = dataset.get_pupil_location()
                except:
                    continue

                # normalize traces per cell to [0, 1]
                traces = max_min_scale(traces, eps = 1e-12)

                # find out where to index the traces and pupil locations
                for i_trial in range(n_trials):
                    stim_table_trial = stim_table[stim_table['repeat'] == i_trial]
                    start_ind = stim_table_trial.iloc[0]['end']
                    n_frames = stim_table_trial.iloc[-1]['frame'] + 1
                    end_ind = start_ind + n_frames

                    # index the pupil locations and traces based on the trial
                    pupil_locs_trial = pupil_locs[start_ind:end_ind, :]
                    traces_trial = traces[0, start_ind:end_ind]

                    # filter out data with missing eye locations at a proportion greater than missing_pupil_coords_thresh
                    missing_prop_pupil = np.mean(np.isnan(pupil_locs_trial[:, 0]))
                    if args.missing_pupil_coords_thresh and (missing_prop_pupil > args.missing_pupil_coords_thresh):
                        traces_trial[:] = np.nan
                        pupil_locs_trial[:, :] = np.nan

                    # change the pupil locations to fractions between 0 and 1 so resizing the image won't change it
                    pupil_locs_trial[:, 0] = (pupil_locs_trial[:, 0] + 608 // 2) / 608
                    pupil_locs_trial[:, 1] = (pupil_locs_trial[:, 1] + 304 // 2) / 304

                    # append this data to the existing files
                    if cell_count == 0 and i_trial == 0:
                        header = ['container/experiment/cell_id/cell_ind']
                        header += [frame_name + '_{}.png'.format(trial_num) for trial_num in range(n_trials) for frame_name in get_img_frame_names(n_frames)]
                        write_csv([header], save_fpath_stimulus_traces, mode = 'a')
                        write_csv([header], save_fpath_stimulus_pupil, mode = 'a')

                    if i_trial == 0:
                        traces_cell.append('{}/{}/{}/{}'.format(container, experiment, cell_id, cell_ind))
                        pupil_coords_cell.append('{}/{}/{}/{}'.format(container, experiment, cell_id, cell_ind))

                    # add the data from this trial to the data for that cell
                    traces_cell += [round(float(trace), 2) for trace in traces_trial]
                    pupil_coords_cell += ['{}/{}'.format(round(float(x), 2), round(float(y), 2)) for x,y in pupil_locs_trial]

                # write data from that cell to the file
                write_csv([traces_cell], save_fpath_stimulus_traces, mode = 'a')
                write_csv([pupil_coords_cell], save_fpath_stimulus_pupil, mode = 'a')

                cell_count += 1



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
    parser.add_argument('manifest_path',
        type = str,
        help = 'Path to the manifest file.')
    parser.add_argument('experiment_dir',
        type = str,
        help = 'Path to the folder called ophys_experiment_data.')
    parser.add_argument('save_dir',
        type = str,
        help = 'Directory to put the extracted data.')
    parser.add_argument('--stimuli',
        type = str,
        nargs = '+',
        help = 'Stimuli to save templates and responses for.')
    parser.add_argument('--n_workers',
        type = int,
        default = 4,
        help = 'Number of workers to use. Default = 4')
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

    # trace and pupil coords args
    parser.add_argument('--missing_pupil_coords_thresh',
        type = float,
        help = 'Will not save data files if the fraction of missing pupil coords is \
            above this value. Should be in the range (0, 1).')

    # receptive field args
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
        save_stimuli(data, args.save_dir, args.n_workers, args.stimuli)

    if args.save_traces_and_pupil_coords:
        save_traces_and_pupil_data(data, args.save_dir, args.missing_pupil_coords_thresh, args.stimuli)

    if args.save_rfs:
        multiproc(save_receptive_fields, [data, args.rf_alpha, args.save_dir, args.sig_chi_only], n_workers = args.n_workers)
