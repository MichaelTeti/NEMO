from glob import glob
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from oct2py import octave
import pandas as pd
import seaborn
import torch
from torchvision.utils import make_grid

from nemo.data.preprocess.image import max_min_scale
from nemo.data.utils import read_csv
from nemo.model.openpv_utils import read_activity_file



def get_mean_activations(model_activity_fpath, openpv_path = '/home/mteti/OpenPV/mlab/util'):
    '''
    Read in the activations of each neuron, average them over space and batch,
    sort them based on the average.
    
    Args:
        model_activity_fpath (str): Path to the <model_layer_name>_A.pvp where the
            activations of each neuron are contained in.
        openpv_path (str): Path to the openpv matlab utility directory (*/OpenPV/mlab/util).
        
    Returns: 
        sorted_acts (np.ndarray): A 4D of shape Batch x Input H x Input W x # Neurons,
            that is sorted descending at the last axis by each neuron's mean activation.
        sorted_mean_acts(np.ndarray): A 1D array of size # Neurons containing each 
            neuron's mean activation and sorted descending by this value.
        sorted_inds (list): A list of length n_neurons corresponding to sorted_mean_acts and 
            consisting of the original index of the neuron with that sorted activation.
    '''
    
    # read in the activations
    acts = read_activity_file(model_activity_fpath, openpv_path = openpv_path)

    # get the mean activation for each neuron across batch and spatial dimensions
    # and sort them descending based on this mean and keep the neuron index too
    mean_acts = np.mean(acts, (0, 1, 2))
    se_acts = np.std(acts, (0, 1, 2)) / np.sqrt(acts[..., 0].size)
    inds = list(range(mean_acts.size))
    sorted_inds = [ind for _,ind in sorted(zip(mean_acts, inds), reverse = True)]
    sorted_mean_acts = mean_acts[sorted_inds]
    sorted_se_acts = se_acts[sorted_inds]
    
    return sorted_mean_acts, sorted_se_acts, sorted_inds


def write_complex_cell_strfs(weight_tensors, write_fpath = 'features.gif', sort_inds = None):
    '''
    View complex cell (spatially-shared weights) strfs in a grid.
    
    Args:
        weight_tensors (list): A list of np.ndarrays of shape out_c x in_c x kh x kw.
        write_fpath (str): The file path to save the resulting .gif as. The parent
            directory should already exist.
        sort_inds (list): List of indices of length out_c to sort the features by in the grid.
        openpv_path (str): Path to the openpv matlab utility directory (*/OpenPV/mlab/util).
        
    Returns:
        None
    '''
    
    n_frames = len(weight_tensors) # number of video frames in the input
        
    # loop through each weight file, extract the weights, and aggregate them
    for frame_num, weights in enumerate(weight_tensors):
        n_neurons, in_c, patch_height, patch_width = weights.shape
        
        # sort the features based on the activation indices if given
        if sort_inds:
            weights = weights[sort_inds]
        
        # make the grid for the features corresponding to this video frame
        grid = make_grid(
            torch.from_numpy(weights), 
            nrow = int(np.floor(np.sqrt(n_neurons))),
            normalize = False,
            scale_each = False,
            padding = 0
        )
        grid = grid.numpy().transpose([1, 2, 0]) # need to put channels last again
        
        # aggregate grids in grid (and make placeholder for grids if first frame)
        if frame_num == 0:
            grids = np.zeros([n_frames] + list(grid.shape))
        
        grids[frame_num] = grid
        
    # scale values in the grid to [0, 255]
    grids = np.uint8(max_min_scale(grids) * 255)
    
    # put a black border in between the features
    grids[:, ::patch_height, ...] = 0.0
    grids[:, :, ::patch_width, :] = 0.0
    
    # write the .gif
    imageio.mimwrite(
        write_fpath,
        [grids[frame_num] for frame_num in range(n_frames)]
    )
    
    
def write_simple_cell_strfs(weight_tensors, save_dir):
    '''
    View simple cell (no spatial sharing of weights) strfs in a grid.
    
    Args:
        weight_tensors (list): List of np.ndarrays of shape n_neurons x in_c x 
            n_features_y x n_features_x x kh x kw.
        save_dir (str): The directory to save the feature grids in, since there will
            be multiple grids.
        
    Returns:
        None
    '''
    
    # make sure save_dir exists or create it
    os.makedirs(save_dir, exist_ok = True)

    n_frames = len(weight_tensors)

    for frame_num, weights in enumerate(weight_tensors): 
        n_neurons, w_in, n_features_y, n_features_x, w_y, w_x = weights.shape

        for neuron_num, weights_feat in enumerate(weights):
            # reshape to n_features_x * n_features_y x in_c x kh x kw
            weights_feat = weights_feat.transpose([1, 2, 0, 3, 4])
            weights_feat = weights_feat.reshape([-1, w_in, w_y, w_x])
            
            # make an image grid
            grid = make_grid(
                torch.from_numpy(weights_feat),
                nrow = n_features_x,
                normalize = False,
                scale_each = False,
                padding = 0
            )
            grid = grid.numpy().transpose([1, 2, 0])
            
            # aggregate grids in grid (and make placeholder for grids if first frame)
            if frame_num == 0 and neuron_num == 0:
                grids = np.zeros([n_neurons, n_frames] + list(grid.shape))
                
            grids[neuron_num, frame_num] = grid
            
    # scale all values to [0, 255]
    grids = np.uint8(max_min_scale(grids) * 255)
    
    # add makeshift black border between feats
    grids[:, :, ::w_y, :, :] = 0.0
    grids[:, :, :, ::w_x, :] = 0.0
    
    # save the grids per feature
    for neuron_num in range(n_neurons):
        imageio.mimwrite(
           os.path.join(save_dir, 'feature{}.gif'.format(neuron_num)),
           [grids[neuron_num, frame_num] for frame_num in range(n_frames)]
        )
        

def get_mean_sparsity(model_activity_fpath, openpv_path = '/home/mteti/OpenPV/mlab/util'):
    '''
    Read in the activations of each neuron and compute proportion inactive per neuron
        over the batch and spatial dimensions.
    
    Args:
        model_activity_fpath (str): Path to the <model_layer_name>_A.pvp where the
            activations of each neuron are contained in.
        openpv_path (str): Path to the openpv matlab utility directory (*/OpenPV/mlab/util).
        
    Returns: 
        sorted_mean_sparsity(np.ndarray): A 1D array of size # Neurons containing each 
            neuron's mean sparsity and sorted descending by this value.
        sorted_inds (list): A list of length n_neurons corresponding to sorted_mean_sparsity and 
            consisting of the original index of the neuron with that sorted mean sparsity.
    '''
    
    # read in activations
    acts = read_activity_file(model_activity_fpath, openpv_path = openpv_path)
    
    # calculate mean sparsity and sort
    mean_sparsity = np.mean(acts != 0.0, (0, 1, 2))
    inds = list(range(mean_sparsity.size))
    sorted_inds = [ind for _,ind in sorted(zip(mean_sparsity, inds), reverse = True)]
    sorted_mean_sparsity = mean_sparsity[sorted_inds]
    
    return sorted_mean_sparsity, sorted_inds


def get_percent_neurons_active(sparse_activity_fpath, n_neurons, feat_map_h, feat_map_w,
        openpv_path = '/home/mteti/OpenPV/mlab/util'):
    '''
    Computes the mean number of neurons active over each batch. 
    
    Args:
        sparse_activity_fpath (str): The <model_layer>.pvp file that contains the activities in 
            sparse format (e.g. S1.pvp).
        n_neurons (int): The number of neurons (i.e. number of features in the dictionary).
        feat_map_h (int): The height of the feature map. 
        feat_map_w (int): The width of the feature map. 
        openpv_path (str): Path to the openpv matlab utility directory (*/OpenPV/mlab/util).
        
    Returns:
        means (np.ndarray): A 1D array of size equal to the number of checkpoints contained
            in the sparse_activity_file and representing the mean number of neurons active at 
            each checkpoint over all batch samples.
        ses (np.ndarray): A 1D array of size equal to the number of checkpoints contained in 
            the sparse_activity_file and representing the standard error of neurons active 
            at each checkpoint over all batch samples corresponding to means.
    '''
    
    # add OpenPV matlab utility dir to octave path
    octave.addpath(openpv_path)
    
    # read in the file given by sparse_activity_fpath
    acts = octave.readpvpfile(sparse_activity_fpath)
    n_display_periods = len(set([act['time'] for act in acts]))
    n_batch = len(acts) // n_display_periods
    sqrt_n = np.sqrt(n_batch)
    total_acts = n_neurons * feat_map_h * feat_map_w
    means, ses = np.zeros([n_display_periods]), np.zeros([n_display_periods])
    
    for display_period in range(n_display_periods):
        acts_over_batch = acts[display_period * n_batch: (display_period + 1) * n_batch]
        n_active_over_batch = [sample_acts['values'].shape[0] / total_acts * 100 for sample_acts in acts_over_batch]
        means[display_period] = np.mean(np.array(n_active_over_batch, dtype = np.float64))
        ses[display_period] = np.std(np.array(n_active_over_batch, dtype = np.float64)) / sqrt_n
        
    return means, ses


def plot_objective_probes(probe_dir, save_dir, probe_type, probe_key, 
        display_period = 3000, n_display_periods = None, plot_individual = False):
    '''
    Plot L2, Total Energy, and L1 Probe data written out by PetaVision.
    
    Args:
        probe_dir (str): Directory containing the .txt probe files.
        save_dir (str): Directory to save the plots.
        probe_type (str): 'energy', 'firm_thresh', or 'l2' for whichever probe
            you are plotting. This is needed because they have different formats.
        probe_key (str): A key to help select and differentiate the desired probe
            files from all other probe files in probe_dir. For example, something
            like "EnergyProbe*" if probe_type is energy.
        n_display_periods (int): How many display periods (starting at the end and 
            moving backwards) to plot here. If not given, will plot them all.
        display_period (int): Number of timesteps in the display period. This is 
            important if n_display_periods is given and for the moving average
            plot. 
        plot_individual (bool): If True, will plot a graph for each probe .txt file.
            
    Returns:
        None
    '''
    
    if probe_type not in ['energy', 'firm_thresh', 'l2']:
        raise ValueError("probe_type not in ['energy', 'firm_thresh', 'l2'].")
    
    # make sure save_dir exists or create it
    os.makedirs(save_dir, exist_ok = True)
    
    # get a list of the probe files
    probe_fpaths = glob(os.path.join(probe_dir, probe_key))
    
    # make an empty dataframe for aggregating across samples for moving average plotting
    probe_agg = pd.DataFrame(columns = ['Timestep', 'ProbeVal'])

    # read in and plot each probe file at a time
    for probe_fpath in probe_fpaths:
        probe = pd.read_csv(
            probe_fpath, 
            usecols = [0, 2] if probe_type == 'energy' else [0, 3],
            header = 0 if probe_type == 'energy' else 'infer',
            names = ['Timestep', 'ProbeVal']
        )
        probe = probe[probe['Timestep'] > 0]
        
        # keep only last value in each display period and add to the probe_agg df for moving avg plotting
        probe_end_val = probe[probe['Timestep'] % display_period == 0]
        probe_agg = probe_agg.append(probe_end_val, ignore_index = True)
        
        if plot_individual:
            # only plot n_display_periods of information since it can be hard to see
            if n_display_periods: 
                probe = probe[probe['Timestep'] > probe['Timestep'].max() - n_display_periods * display_period]

            # plot the data for this probe file
            seaborn.lineplot(data = probe, x = 'Timestep', y = 'ProbeVal')

            # make ylabel depending on the probe_type 
            if probe_type == 'energy':
                plt.ylabel('Total Energy')
            elif probe_type == 'firm_thresh':
                plt.ylabel('L1 Sparsity Value')
            else:
                plt.ylabel('L2 Reconstruction Error')

            # save the figure
            fig_fname = os.path.split(probe_fpath)[1]
            plt.savefig(
                os.path.join(save_dir, os.path.splitext(fig_fname)[0] + '.png'),
                bbox_inches = 'tight'
            )
            plt.close()
        
        
    # plot the moving average
    probe_agg['Display Period'] = probe_agg['Timestep'] // display_period
    seaborn.lineplot(data = probe_agg, x = 'Display Period', y = 'ProbeVal')
    
    # make y label depending on probe type
    if probe_type == 'energy':
        plt.ylabel('Final Energy Value in Display Period (95% CI Shaded)')
    elif probe_type == 'firm_thresh':
        plt.ylabel('Final L1 Sparsity Value in Display Period (95% CI Shaded)')
    else:
        plt.ylabel('Final L2 Reconstruction Error in Display Period (95% CI Shaded)')
    
    plt.savefig(os.path.join(save_dir, 'final_probe_val.png'), bbox_inches = 'tight')
    plt.close()
    
    
def plot_adaptive_timescale_probes(probe_dir, save_dir, probe_key, display_period = 3000, 
        n_display_periods = None, plot_individual = False):
    '''
    Plot data in adaptive timescale probe. 
    
    Args:
        probe_dir (str): Directory containing the .txt probe files.
        save_dir (str): Directory to save the plots.
        probe_key (str): A key to help select and differentiate the desired probe
            files from all other probe files in probe_dir. For example, something
            like "EnergyProbe*" if probe_type is energy.
        n_display_periods (int): How many display periods (starting at the end and 
            moving backwards) to plot here. If not given, will plot them all.
        display_period (int): Number of timesteps in the display period. This is 
            important if n_display_periods is given and for the moving average
            plot.
        plot_individual (bool): If True, will plot a graph for each probe .txt file.
            
    Returns:
        None
    '''
    
    # make sure save_dir exists or create it
    os.makedirs(save_dir, exist_ok = True)
    
    ts_dir = os.path.join(save_dir, 'Timescale')
    os.makedirs(ts_dir, exist_ok = True)
    
    ts_max_dir = os.path.join(save_dir, 'TimescaleMax')
    os.makedirs(ts_max_dir, exist_ok = True)
    
    ts_true_dir = os.path.join(save_dir, 'TimescaleTrue')
    os.makedirs(ts_true_dir, exist_ok = True)

    # get a list of the probe files
    probe_fpaths = glob(os.path.join(probe_dir, probe_key))
    
    # make an empty dataframe for aggregating across samples for moving average plotting
    probe_agg = pd.DataFrame(columns = ['Timestep', 'Timescale', 'TimescaleTrue', 'TimescaleMax'])

    for fpath in probe_fpaths:
        # adaptive timescale probes are saved differently, so need to parse out the values
        data = read_csv(fpath)
        times = [float(row.split(' = ')[1]) for row in data[0::2]]
        ts = [float(row[1].split(' = ')[1]) for row in data[1::2]]
        ts_true = [float(row[2].split(' = ')[1]) for row in data[1::2]]
        ts_max = [float(row[3].split(' = ')[1]) for row in data[1::2]]
        
        # create a dataframe to put all of these probe values
        probe = pd.DataFrame(
            data = {
                'Timestep': times,
                'Timescale': ts,
                'TimescaleTrue': ts_true,
                'TimescaleMax': ts_max
            }
        )
        
        # keep only last value in each display period and add to the probe_agg df for moving avg plotting
        probe_end_val = probe[probe['Timestep'] % display_period == 0]
        probe_agg = probe_agg.append(probe_end_val, ignore_index = True)
        
        if plot_individual:
            # only plot n_display_periods of information since it can be hard to see
            if n_display_periods: 
                probe = probe[probe['Timestep'] > probe['Timestep'].max() - n_display_periods * display_period]

            # make name for each figure based on the .txt file name
            fig_fname = os.path.splitext(os.path.split(fpath)[1])[0] + '.png'
            
            # plot and save each probe
            seaborn.lineplot(data = probe, x = 'Timestep', y = 'Timescale')
            plt.savefig(os.path.join(ts_dir, fig_fname), bbox_inches = 'tight')
            plt.close()

            seaborn.lineplot(data = probe, x = 'Timestep', y = 'TimescaleTrue')
            plt.savefig(os.path.join(ts_true_dir, fig_fname), bbox_inches = 'tight')
            plt.close()

            seaborn.lineplot(data = probe, x = 'Timestep', y = 'TimescaleMax')
            plt.savefig(os.path.join(ts_max_dir, fig_fname), bbox_inches = 'tight')
            plt.close()
            
        
    # plot the moving averages
    probe_agg['Display Period'] = probe_agg['Timestep'] // display_period
        
    seaborn.lineplot(data = probe_agg, x = 'Display Period', y = 'Timescale')
    plt.ylabel('Final Timescale Value in Display Period (95% CI Shaded)')
    plt.savefig(os.path.join(ts_dir, 'final_probe_val.png'), bbox_inches = 'tight')
    plt.close()

    seaborn.lineplot(data = probe_agg, x = 'Display Period', y = 'TimescaleTrue')
    plt.ylabel('Final Timescale True Value in Display Period (95% CI Shaded)')
    plt.savefig(os.path.join(ts_true_dir, 'final_probe_val.png'), bbox_inches = 'tight')
    plt.close()

    seaborn.lineplot(data = probe_agg, x = 'Display Period', y = 'TimescaleMax')
    plt.ylabel('Final Timescale Max Value in Display Period (95% CI Shaded)')
    plt.savefig(os.path.join(ts_max_dir, 'final_probe_val.png'), bbox_inches = 'tight')
    plt.close()