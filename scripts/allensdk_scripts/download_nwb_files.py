from argparse import ArgumentParser
import os, sys

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info

from NEMO.utils.general_utils import multiproc


def download_experiment_data(ids, boc):
    '''
    Download AllenSDK experiment container files.
    Args:
        ids (list): experiment ids to download data.
        boc (BrainObservatoryCache object)
    Returns:
        None

    '''

    for id in ids:
        boc.get_ophys_experiment_data(id)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--manifest_save_dir',
        type = str,
        help = 'Where to save the manifest file. Default is in the current directory.')
    parser.add_argument('--n_workers',
        type = int,
        help = 'Number of CPUs to use when downloading NWB files. Default is 1.')
    parser.add_argument('--n_experiments',
        type = int,
        help = 'Number of experiments to download. Default is all of them.')

    # filter args for experiments / containers
    parser.add_argument('--targeted_structures',
        type = str,
        nargs = '+',
        help = 'Targeted structures. Default is all of them.')
    parser.add_argument('--cre_lines',
        type = str,
        nargs = '+',
        help = 'Desired Cre lines. Default is all of them.')
    parser.add_argument('--imaging_depths',
        type = int,
        nargs = '+',
        help = 'Desired imaging depths. Default is all of them.')
    parser.add_argument('--session_type',
        type = str,
        nargs = '+',
        help = 'Choose a specific session type to pull. Session types include \
            session_three_A, session_three_B, session_three_C, session_three_C2.')
    args = parser.parse_args()


    manifest_path = 'manifest.json' if not args.manifest_save_dir else os.path.join(args.manifest_save_dir, 'manifest.json')
    boc = BrainObservatoryCache(manifest_file = manifest_path)

    # get experiment containers
    conts = boc.get_experiment_containers(targeted_structures = args.targeted_structures,
                                          imaging_depths = args.imaging_depths,
                                          include_failed = False)

    # filter based on Cre line
    if args.cre_lines:
        conts = [cont for cont in conts if any([cre_line in cont['specimen_name'] for cre_line in args.cre_lines])]

    # filter out ones with eplieptiform events
    conts = [cont for cont in conts if 'Epileptiform Events' not in cont['tags']]

    # get experiments from the containers
    exps = boc.get_ophys_experiments(experiment_container_ids = [cont['id'] for cont in conts])

    # remove the experiments that failed the eye tracking
    exps = [exp for exp in exps if not exp['fail_eye_tracking']]

    # limit number of experiments to download if specified
    if args.n_experiments: exps = exps[:args.n_experiments]

    # get specific experiments with both locally-sparse-noise and natural movies
    if args.session_type:
        exps = [exp for exp in exps if exp['session_type'] in args.session_type]

    print(['[INFO] FOUND {} EXPERIMENTS'.format(len(exps))])

    # download experiment data based on id
    exp_ids = [exp['id'] for exp in exps]
    download_data_args = [exp_ids, boc]
    multiproc(download_experiment_data, download_data_args, n_workers = args.n_workers)
