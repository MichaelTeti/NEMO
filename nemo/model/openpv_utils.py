from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import oct2py
from oct2py import octave
import pandas as pd
import seaborn

from nemo.data.utils import read_csv



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


def read_activity_file(model_activity_fpath, openpv_path = '/home/mteti/OpenPV/mlab/util'):
    '''
    Read in the <model_layer_name>_A.pvp file containing the activations / feature maps of each neuron.
    
    Args:
        model_activity_fpath (str): Path to the <model_layer_name>_A.pvp where the
            activations of each neuron are contained in.
        openpv_path (str): Path to the openpv matlab utility directory (*/OpenPV/mlab/util).
        
    Returns:
        acts (np.ndarray): A B x H x W x # Neurons array of feature maps. 
    '''
    
    # add OpenPV matlab utility dir to octave path
    octave.addpath(openpv_path)
    
    # Read in the activity file and get the number of batch samples
    # Should be a list of length batch size, where each item in the list is an array 
    # of shape (input_width / stride x) x (input height / stride y) x # Neurons
    # with the activations for that particular batch sample
    act_data = octave.readpvpfile(model_activity_fpath)
    n_batch = len(act_data)
    
    # concatenate across batch samples to produce a single array of shape 
    # batch size x (input width / stride x) x (input height / stride y) x # Neurons
    acts = np.concatenate([act_data[b]['values'][None, ...] for b in range(n_batch)], 0)
    
    # transpose the 2nd and third dimensions to make it B x H x W x # Neurons
    acts = acts.transpose([0, 2, 1, 3])
    
    return acts


def read_complex_cell_weight_files(fpaths, openpv_path = '/home/mteti/OpenPV/mlab/util'):
    '''
    Reads in .pvp shared weights files with convolutional features. 

    Args:
        fpaths (list): List of file paths corresponding to the .pvp weight files to read.
        openpv_path (str): Path to */OpenPV/mlab/util. 

    Returns: 
        weights_agg (list): List of len(fpaths) arrays of shape n_neurons x in_c x kh x kw.
    '''

    octave.addpath(openpv_path)

    weights_agg = []
    for fpath in fpaths:
        # original shape: patch width x patch height x in channels x out channels
        # reshape to: out_channels x in channels x patch height x patch width
        weights = octave.readpvpfile(fpath)[0]['values'][0]
        weights = weights.transpose([3, 2, 1, 0])
        weights_agg.append(weights)

    return weights_agg


def read_input_and_recon_files(fpaths, openpv_path = '/home/mteti/OpenPV/mlab/util'):
    '''
    Reads inputs and recons from .pvp files. 

    Args:
        fpaths (list): List of filepaths to read data from.
        openpv_path (str): Path to */OpenPV/mlab/util.

    Returns:
        data_agg (np.ndarray): Array of shape B x F x H x W, where F is len(fpaths).
    '''

    octave.addpath(openpv_path)

    data_agg = []
    for fpath_num, fpath in enumerate(fpaths):
        batch = [sample['values'] for sample in octave.readpvpfile(fpath)]
        data_agg.append(batch)

    return np.asarray(data_agg).transpose([1, 0, 3, 2])


def read_simple_cell_weight_files(fpaths, n_features_x, n_features_y, 
                                  openpv_path = '/home/mteti/OpenPV/mlab/util'):
    '''
    Reads in .pvp shared weight files, where the features are the same size as the inputs,
    and stride_x and stride_y equal the input width and height, respectively. 

    Args:
        fpaths (list): List of file paths corresponding to the .pvp weight files to read.
        n_features_x (int): Number of simple cell features to display along grid width.
        n_features_y (int): Number of simple cell features to display down grid height. 
        openpv_path (str): Path to */OpenPV/mlab/util. 

    Returns:
        weights_agg (list): List of len(fpaths) arrays of shape n_neurons x in_c x 
            n_features_y x n_features_x x kh x kw. 
    '''

    octave.addpath(openpv_path)

    weights_agg = []
    for fpath in fpaths:
        weights = octave.readpvpfile(fpath)[0]['values'][0]
        w_x, w_y, w_in, w_out = weights.shape
        n_neurons = w_out // n_features_x // n_features_y
        
        # reshape to the original shape of the unshared weight tensor 
        # and reverse the x and y dims for image writing
        weights = weights.reshape([w_x, w_y, w_in, n_neurons, n_features_x, n_features_y])
        weights = weights.transpose([3, 2, 5, 4, 1, 0])
        weights_agg.append(weights)

    return weights_agg


def np_to_pvp_shared_weight_file(tensors, fpaths, openpv_path):
    ''' 
    Writes numpy arrays as a pvp shared weight files.
    
    Args:
        tensors (list): np.ndarrays to write to pvp shared weight files.
        fpaths (list): File paths corresponding to the write location of each tensor.
        openpv_path (str): Path to */OpenPV/mlab/util.
        
    Returns:
        None
    '''
    
    oct2py.octave.addpath(openpv_path)
    for file_num, (tensor, fpath) in enumerate(zip(tensors, fpaths)):
        os.makedirs(os.path.split(fpath)[0], exist_ok = True)
        data = [{'time': 0.0, 'values': [tensor]}]
        oct2py.octave.push(['data', 'fpath'], [data, fpath])
        oct2py.octave.eval('writepvpsharedweightfile(fpath, data)')