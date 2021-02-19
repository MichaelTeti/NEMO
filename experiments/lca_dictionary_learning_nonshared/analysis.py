from argparse import ArgumentParser
import logging
import os

import matplotlib.pyplot as plt

from nemo.model.analysis.lca import (
    get_mean_activations,
    get_mean_sparsity,
    view_simple_cell_strfs,
    get_percent_neurons_active,
    plot_objective_probes,
    plot_adaptive_timescale_probes,
    view_reconstructions,
    read_activity_file
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

# plotting the number of neurons active at once
n_active_args = parser.add_argument_group(
    'number active',
    description = 'The arguments for the plot_num_neurons_active function.'
)
n_active_args.add_argument(
    '--sparse_activity_fpath',
    required = True,
    type = str,
    help = 'Path to the <model_layer_name>.pvp file.'
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


if not args.no_features:
    logging.info('WRITING FEATURES')
    view_simple_cell_strfs(
        ckpt_dir = args.ckpt_dir,
        save_dir = os.path.join(args.save_dir, 'Features'),
        n_features_y = 8,
        n_features_x = 24,
        weight_file_key = args.weight_fpath_key,
        openpv_path = args.openpv_path
    )

if not args.no_recons:
    logging.info('WRITING INPUTS AND RECONSTRUCTIONS')
    view_reconstructions(
        ckpt_dir = args.ckpt_dir,
        save_dir = os.path.join(args.save_dir, 'Inputs_and_Recons'),
        recon_layer_key = args.rec_layer_key,
        input_layer_key = args.input_layer_key,
        openpv_path = args.openpv_path
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
    mean, se, _ = get_mean_activations(args.activity_fpath, openpv_path = args.openpv_path)
    plt.errorbar(x = list(range(mean.size)), y = mean, yerr = se)
    plt.xlabel('Neuron Index')
    plt.ylabel('Mean Activation +/- 1 SE')
    plt.savefig(os.path.join(args.save_dir, 'mean_activations.png'), bbox_inches = 'tight')
    plt.close()

    mean_sparsity, _ = get_mean_sparsity(args.activity_fpath, openpv_path = args.openpv_path)
    plt.plot(mean_sparsity)
    plt.xlabel('Neuron Index')
    plt.ylabel('Mean Sparsity')
    plt.savefig(os.path.join(args.save_dir, 'mean_sparsity.png'), bbox_inches = 'tight')
    plt.close()

    _, feat_map_h, feat_map_w, n_neurons = read_activity_file(args.activity_fpath).shape
    mean_active, se_active = get_percent_neurons_active(
        args.sparse_activity_fpath, 
        n_neurons,
        feat_map_h,
        feat_map_w,
        openpv_path = args.openpv_path
    )
    plt.errorbar(x = list(range(mean_active.size)), y = mean_active, yerr = se_active)
    plt.xlabel('Display Period Number')
    plt.ylabel('Mean Percent Active Over Batch +/- 1 SE')
    plt.savefig(os.path.join(args.save_dir, 'mean_percent_active.png'), bbox_inches = 'tight')
    plt.close()
