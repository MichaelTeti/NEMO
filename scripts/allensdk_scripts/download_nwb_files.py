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
    parser.add_argument(
        '--manifest_save_dir',
        type = str,
        help = 'Where to save the manifest file. Default is in the current directory.'
    )
    parser.add_argument(
        '--n_workers',
        type = int,
        help = 'Number of CPUs to use when downloading NWB files. Default is 1.'
    )
    parser.add_argument(
        '--n_experiments',
        type = int,
        help = 'Number of experiments to download. Default is all of them.'
    )

    # filter args for experiments / containers
    parser.add_argument(
        '--targeted_structures',
        type = str,
        nargs = '+',
        help = 'Targeted structures. Default is all of them.'
    )
    parser.add_argument(
        '--cre_lines',
        type = str,
        nargs = '+',
        choices = [
            'Cux2-CreERT2',
            'Emx1-IRES-Cre',
            'Fezf2-CreER',
            'Nr5a1-Cre',
            'Ntsr1-Cre_GN220',
            'Pvalb-IRES-Cre',
            'Rbp4-Cre_KL100',
            'Rorb-IRES2-Cre',
            'Scnn1a-Tg3-Cre',
            'Slc17a7-IRES2-Cre',
            'Sst-IRES-Cre',
            'Tlx3-Cre_PL56',
            'Vip-IRES-Cre'
        ],
        help = 'Desired Cre lines. Default is all of them.'
    )
    parser.add_argument(
        '--reporter_lines',
        type = str,
        nargs = '+',
        choices = [
            'Ai148(TIT2L-GC6f-ICL-tTA2)',
            'Ai162(TIT2L-GC6s-ICL-tTA2)',
            'Ai93(TITL-GCaMP6f)',
            'Ai93(TITL-GCaMP6f)-hyg',
            'Ai94(TITL-GCaMP6s)'
        ],
        help = 'Desired reporter lines. Default is all of them.'
    )
    parser.add_argument(
        '--imaging_depths',
        type = int,
        nargs = '+',
        help = 'Desired imaging depths. Default is all of them.'
    )
    parser.add_argument(
        '--session_type',
        type = str,
        nargs = '+',
        help = 'Choose a specific session type to pull. Session types include \
            session_three_A, session_three_B, session_three_C, session_three_C2.'
    )
    args = parser.parse_args()


    manifest_path = 'manifest.json' if not args.manifest_save_dir else os.path.join(args.manifest_save_dir, 'manifest.json')
    boc = BrainObservatoryCache(manifest_file = manifest_path)

    # get experiment containers, which have data for a specific
    # target area, imaging depth, and cre line
    conts = boc.get_experiment_containers(
        targeted_structures = args.targeted_structures,
        imaging_depths = args.imaging_depths,
        include_failed = False,
        cre_lines = args.cre_lines,
        reporter_lines = args.reporter_lines
    )

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
    multiproc(
        func = download_experiment_data,
        iterator_key = 'ids',
        n_workers = args.n_workers,
        ids = exp_ids,
        boc = boc
    )

# python3 download_nwb_files.py \
#     --manifest_save_dir ../../../BrainObservatoryData \
#     --n_workers 4 \
#     --cre_lines "Rorb-IRES2-Cre" "Scnn1a-Tg3-Cre" "Nr5a1-Cre" \
#     --reporter_lines "Ai93(TITL-GCaMP6f)" \
#     --targeted_structures "VISp"
