from argparse import ArgumentParser
import os, sys

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info

from nemo.data.utils import multiproc, download_experiment_data


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
    choices = [
        'VISp',
        'VISl',
        'VISal',
        'VISrl',
        'VISam',
        'VISpm'
    ],
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
    '--min_imaging_depth',
    type = int,
    help = 'Minimum desired imaging depth.'
    # http://www.nibb.ac.jp/brish/Gallery/cortexE.html
    # https://www.jneurosci.org/content/35/18/7287
    # https://sci-hub.se/10.3791/60600 250-450um
)
parser.add_argument(
    '--max_imaging_depth',
    type = int,
    help = 'Maximium desired imaging depth.'
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
    include_failed = False,
    cre_lines = args.cre_lines,
    reporter_lines = args.reporter_lines
)

# filter out ones with eplieptiform events
conts = [cont for cont in conts if 'Epileptiform Events' not in cont['tags']]

# filter imaging depths here so we can just give the range 
if args.min_imaging_depth:
    conts = [cont for cont in conts if cont['imaging_depth'] >= args.min_imaging_depth]
if args.max_imaging_depth:
    conts = [cont for cont in conts if cont['imaging_depth'] <= args.max_imaging_depth]

# get experiments from the containers
exps = boc.get_ophys_experiments(experiment_container_ids = [cont['id'] for cont in conts])

# remove the experiments that failed the eye tracking
exps = [exp for exp in exps if not exp['fail_eye_tracking']]

# limit number of experiments to download if specified
if args.n_experiments: exps = exps[:args.n_experiments]

# get specific experiments with both locally-sparse-noise and natural movies
if args.session_type:
    exps = [exp for exp in exps if exp['session_type'] in args.session_type]

print('[INFO] FOUND {} EXPERIMENTS'.format(len(exps)))

# download experiment data based on id
exp_ids = [exp['id'] for exp in exps]
multiproc(
    func = download_experiment_data,
    iterator_keys = ['ids'],
    n_workers = args.n_workers,
    ids = exp_ids,
    boc = boc
)
