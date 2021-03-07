from argparse import ArgumentParser
import logging
import os

import matplotlib.pyplot as plt
import seaborn

from nemo.data.utils import get_fpaths_in_dir
from nemo.data.io.image import write_gifs
from nemo.model.analysis.feature_visualization import write_complex_cell_strfs
from nemo.model.analysis.metrics import (
    lifetime_sparsity, 
    mean_activations,
    population_sparsity
)
from nemo.model.openpv_utils import (
    plot_adaptive_timescale_probes,
    plot_objective_probes,
    read_activity_file,
    read_complex_cell_weight_files,
    read_input_and_recon_files
)


parser = ArgumentParser()
parser.add_argument(
    'ckpt_dir',
    type = str,
    help = 'Path to the OpenPV checkpoint.'
)
parser.add_argument(
    'save_dir',
    type = str,
    help = 'Where to save these analyses and write out plots.'
)
parser.add_argument(
    '--openpv_path',
    type = str,
    default = '/home/mteti/OpenPV/mlab/util',
    help = 'Path to *OpenPV/mlab/util',
)
parser.add_argument(
    '--no_features',
    action = 'store_true',
    help = 'If specified, will not plot the features.'
)
parser.add_argument(
    '--no_recons',
    action = 'store_true',
    help = 'If specified, will not plot the reconstructions and inputs.'
)
parser.add_argument(
    '--no_probes',
    action = 'store_true',
    help = 'If specified, will not plot the probes.'
)
parser.add_argument(
    '--no_activity',
    action = 'store_true',
    help = 'If specified, will not plot mean activations, mean sparsity, \
        or mean activity.'
)

# visualize features
feat_args = parser.add_argument_group(
    'feature visualization',
    description = 'Arguments for the view_complex_cell_strfs function.'
)
feat_args.add_argument(
    '--weight_fpath_key',
    type = str,
    default = '*_W.pvp',
    help = 'A key to help find the desired _W.pvp files in the ckpt \
        directory, since there may be multiple in the same one for \
        some models.'
)

# visualize reconstructions
rec_args = parser.add_argument_group(
    'reconstruction visualization',
    description = 'The arguments for the view_reconstructions function.'
)
rec_args.add_argument(
    '--rec_layer_key',
    type = str,
    default = 'Frame[0-9]Recon_A.pvp',
    help = 'A key to help find the .pvp files written out by the reconstruction \
        layer in the checkpoint directory.'
)
rec_args.add_argument(
    '--input_layer_key',
    type = str,
    default = 'Frame[0-9]_A.pvp',
    help = 'A key to help find the .pvp files written out by the input layer \
        in the checkpoint directory.'
)

# plotting mean activations and mean sparsity 
mean_act_args = parser.add_argument_group(
    'mean activations',
    description = 'The arguments to plot the mean activations and mean sparstiy.'
)
mean_act_args.add_argument(
    '--activity_fpath',
    required = True,
    type = str,
    help = 'The path to the <model_layer_name>_A.pvp file.'
)

# probe arguments 
probe_args = parser.add_argument_group(
    'probes',
    description = 'The arguments for the plot_probes.py script.'
)
probe_args.add_argument(
    '--probe_dir',
    required = True,
    type = str,
    help = 'The directory where the probes are written out.'
)
probe_args.add_argument(
    '--l2_probe_key',
    type = str,
    default = '*L2Probe*',
    help = 'A key to help filter out the L2 Probe files in probe_dir \
        (e.g. "*L2Probe*").'
)
probe_args.add_argument(
    '--firm_thresh_probe_key',
    type = str,
    default = 'S1FirmThresh*',
    help = 'A key to help filter out the Firm Thresh files in probe_dir.'
)
probe_args.add_argument(
    '--energy_probe_key',
    type = str,
    default = 'Energy*',
    help = 'A key to help filter out the Energy files in the probe_dir.'
)
probe_args.add_argument(
    '--adaptive_ts_probe_key',
    type = str,
    default = 'Adaptive*',
    help = 'A key to help filter out the adaptive timescale probes in probe_dir.'
)
probe_args.add_argument(
    '--display_period_length',
    type = int,
    default = 3000,
    help = 'The length of the display period.'
)
probe_args.add_argument(
    '--n_display_periods',
    type = int,
    help = 'How many display periods to show starting from the end and going backward.'
)
probe_args.add_argument(
    '--plot_individual_probes',
    action = 'store_true',
    help = 'If specified, will plot each probe .txt file individually as well as \
        moving average.'
)

args = parser.parse_args()

# make the save directory
os.makedirs(args.save_dir, exist_ok = True)

logging.basicConfig(
    format='%(asctime)s -- %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p', 
    level = logging.INFO
)


# need this for features and activity
acts = read_activity_file(args.activity_fpath, openpv_path = args.openpv_path)
mean_acts, se_acts, sorted_inds_by_act = mean_activations(acts)


if not args.no_features:
    logging.info('WRITING FEATURES')
    weight_fpaths = get_fpaths_in_dir(args.ckpt_dir, fname_key = args.weight_fpath_key)
    weight_tensors = read_complex_cell_weight_files(weight_fpaths, openpv_path = args.openpv_path)
    write_complex_cell_strfs(
        weight_tensors = weight_tensors,
        write_fpath = os.path.join(args.save_dir, 'features.gif'),
        sort_inds = sorted_inds_by_act
    )

if not args.no_recons:
    logging.info('WRITING INPUTS AND RECONSTRUCTIONS')
    input_fpaths = get_fpaths_in_dir(args.ckpt_dir, args.input_layer_key)
    recon_fpaths = get_fpaths_in_dir(args.ckpt_dir, args.rec_layer_key)
    inputs = read_input_and_recon_files(input_fpaths)
    recons = read_input_and_recon_files(recon_fpaths)
    write_gifs(
        inputs, 
        os.path.join(args.save_dir, 'Inputs_and_Recons', 'Inputs'),
        scale = True
    )
    write_gifs(
        recons, 
        os.path.join(args.save_dir, 'Inputs_and_Recons', 'Recons'),
        scale = True
    )
    write_gifs(
        inputs - recons, 
        os.path.join(args.save_dir, 'Inputs_and_Recons', 'Diffs'),
        scale = True    
    )

# plotting probes below
if not args.no_probes:
    logging.info('WRITING PROBES')
    plot_objective_probes(
        probe_dir = args.probe_dir, 
        save_dir = os.path.join(args.save_dir, 'EnergyProbe'), 
        probe_type = 'energy',
        probe_key = args.energy_probe_key, 
        display_period = args.display_period_length,
        n_display_periods = args.n_display_periods,
        plot_individual = args.plot_individual_probes
    )
    plot_objective_probes(
        probe_dir = args.probe_dir, 
        save_dir = os.path.join(args.save_dir, 'L2Probe'), 
        probe_type = 'l2',
        probe_key = args.l2_probe_key, 
        display_period = args.display_period_length,
        n_display_periods = args.n_display_periods,
        plot_individual = args.plot_individual_probes
    )
    plot_objective_probes(
        probe_dir = args.probe_dir, 
        save_dir = os.path.join(args.save_dir, 'FirmThreshProbe'), 
        probe_type = 'firm_thresh',
        probe_key = args.firm_thresh_probe_key, 
        display_period = args.display_period_length,
        n_display_periods = args.n_display_periods,
        plot_individual = args.plot_individual_probes
    )    
    plot_adaptive_timescale_probes(
        probe_dir = args.probe_dir,
        save_dir = os.path.join(args.save_dir, 'AdaptiveTimescaleProbe'),
        probe_key = args.adaptive_ts_probe_key,
        display_period = args.display_period_length,
        n_display_periods = args.n_display_periods,
        plot_individual = args.plot_individual_probes
    )

if not args.no_activity:
    logging.info('PLOTTING ACTIVATIONS')

    # mean acts, mean sparsity, and number active
    plt.errorbar(x = list(range(mean_acts.size)), y = mean_acts, yerr = se_acts)
    plt.xlabel('Neuron Index')
    plt.ylabel('Mean Activation +/- 1 SE')
    plt.savefig(os.path.join(args.save_dir, 'mean_activations_line.png'), bbox_inches = 'tight')
    plt.close()

    seaborn.boxplot(y = mean_acts)
    plt.ylabel('Mean Activation')
    plt.savefig(os.path.join(args.save_dir, 'mean_activations_box.png'), bbox_inches = 'tight')
    plt.close()

    
    logging.info('PLOTTING LIFETIME SPARSITY')
    lifetime = lifetime_sparsity(acts)
    lifetime.sort()

    plt.plot(lifetime[::-1])
    plt.xlabel('Neuron Index')
    plt.ylabel('Lifetime Sparsity')
    plt.savefig(os.path.join(args.save_dir, 'lifetime_sparsity_line.png'), bbox_inches = 'tight')
    plt.close()

    seaborn.boxplot(y = lifetime)
    plt.ylabel('Lifetime Sparsity')
    plt.savefig(os.path.join(args.save_dir, 'lifetime_sparsity_box.png'), bbox_inches = 'tight')
    plt.close()


    logging.info('PLOTTING POPULATION SPARSITY')
    population = population_sparsity(acts)
    population.sort()

    plt.plot(population[::-1])
    plt.xlabel('Stimulus Index')
    plt.ylabel('Population Sparsity')
    plt.savefig(os.path.join(args.save_dir, 'population_sparsity_line.png'), bbox_inches = 'tight')
    plt.close()

    seaborn.boxplot(y = population)
    plt.ylabel('Population Sparsity')
    plt.savefig(os.path.join(args.save_dir, 'population_sparsity_box.png'), bbox_inches = 'tight')
    plt.close()