from argparse import ArgumentParser
import os
import sys

from oct2py import octave 

from NEMO.utils.plot_probes import (
    plot_energy,
    plot_adaptive_timescales,
    plot_l2,
    plot_firm_thresh
)


parser = ArgumentParser()
parser.add_argument(
    '--openpv_path',
    type = str,
    required = True,
    help = 'Path to the OpenPV/mlab/util directory.'
)
parser.add_argument(
    '--ckpt_dir',
    required = True,
    type = str,
    help = 'Path to the OpenPV checkpoint.'
)
parser.add_argument(
    '--save_dir',
    required = True,
    type = str,
    help = 'Where to save these analyses and write out plots.'
)
parser.add_argument(
    '--model_layer_name',
    type = str,
    default = 'S1',
    help = 'The name of the model layer.'
)

# visualize features
feat_args = parser.add_argument_group(
    'feature visualization',
    description = 'Arguments for the viz_shared_vid_weights.m function.'
)
feat_args.add_argument(
    '--weight_fpath_key',
    type = str,
    help = 'A key to help find the desired _W.pvp files in the ckpt \
        directory, since there may be multiple in the same one for \
        some models.'
)
feat_args.add_argument(
    '--clip_frame_0',
    action = 'store_true',
    help = 'If specified, will clip the frame 0 feature values with \
        the min/max of the features for all other frames. This has to \
        do with the way we initialize the connection values on video \
        stimuli, and it is helpful to see changes in the first few runs \
        after initialization before the weights from all other frames \
        catch up with frame 0.'
)
feat_args.add_argument(
    '--sorted_feats',
    action = 'store_true',
    help = 'If specified, will display the features in descending order \
        of activation from top left to bottom right.'
)

# visualize reconstructions
rec_args = parser.add_argument_group(
    'reconstruction visualization',
    description = 'The arguments for the viz_reconstructions.m function.'
)
rec_args.add_argument(
    '--rec_layer_key',
    type = str,
    default = 'Frame*Recon_A.pvp',
    help = 'A key to help find the .pvp files written out by the reconstruction \
        layer in the checkpoint directory (e.g. "Frame*Recon_A.pvp" where the \
        asterisk represents the frame number).'
)
rec_args.add_argument(
    '--input_layer_key',
    type = str,
    default = 'Frame*_A.pvp',
    help = 'A key to help find the .pvp files written out by the input layer \
        in the checkpoint directory (e.g. "Frame*_A.pvp" where the asterisk \
        represents the frame number).'
)

# plotting the number of neurons active at once
n_active_args = parser.add_argument_group(
    'number active',
    description = 'The arguments for the plot_num_active.m function.'
)
n_active_args.add_argument(
    '--model_write_fpath',
    required = True,
    type = str,
    help = 'Path to the <model_layer>.pvp file.'
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
    '--firm_thresh_key',
    type = str,
    default = 'S1FirmThresh*',
    help = 'A key to help filter out the Firm Thresh files in probe_dir.'
)
probe_args.add_argument(
    '--energy_key',
    type = str,
    default = 'Energy*',
    help = 'A key to help filter out the Energy files in the probe_dir.'
)
probe_args.add_argument(
    '--adaptive_ts_key',
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

args = parser.parse_args()

# make the save directory
os.makedirs(args.save_dir, exist_ok = True)

# run the octave scripts first
octave.addpath(args.openpv_path)
octave.addpath('../../NEMO/utils/')

print('[INFO] WRITING FEATURES...')
feature_save_path = os.path.join(
    args.save_dir,
    'weights_clipped.gif' if args.clip_frame_0 else 'weights.gif'
)
octave.viz_shared_vid_weights(
    args.ckpt_dir,
    os.path.join(args.save_dir, 'weights.gif'),
    args.weight_fpath_key,
    args.clip_frame_0,
    args.sorted_feats,
    os.path.join(args.ckpt_dir, args.model_layer_name + '_A.pvp')
)

print('[INFO] WRITING INPUTS AND RECONSTRUCTIONS...')
octave.viz_reconstructions(
    args.ckpt_dir,
    os.path.join(args.save_dir, 'Inputs_and_Recons'),
    args.rec_layer_key,
    args.input_layer_key
)

octave.plot_num_active(args.model_write_fpath)
octave.get_mean_acts(os.path.join(args.ckpt_dir, args.model_layer_name + '_A.pvp'))

# plotting probes below
print('[INFO] PLOTTING PROBES...')
plot_energy(
    args.probe_dir, 
    os.path.join(args.save_dir, 'EnergyProbe'), 
    args.energy_key, 
    display_period = args.display_period_length,
    n_display_periods = args.n_display_periods
)
plot_adaptive_timescales(
    args.probe_dir, 
    os.path.join(args.save_dir, 'AdaptiveTimescaleProbe'), 
    args.adaptive_ts_key,
    display_period = args.display_period_length,
    n_display_periods = args.n_display_periods
)
plot_firm_thresh(
    args.probe_dir, 
    os.path.join(args.save_dir, 'FirmThreshProbe'), 
    args.firm_thresh_key,
    display_period = args.display_period_length,
    n_display_periods = args.n_display_periods
)
plot_l2(
    args.probe_dir,
    os.path.join(args.save_dir, 'L2Probe'),
    args.l2_probe_key,
    display_period = args.display_period_length,
    n_display_periods = args.n_display_periods
)
