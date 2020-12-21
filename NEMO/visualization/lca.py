from glob import glob
import os

import imageio
import numpy as np
from oct2py import octave
import torch
from torchvision.utils import make_grid

from NEMO.utils.image_utils import max_min_scale


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
    
    # add OpenPV matlab utility dir to octave path
    octave.addpath(openpv_path)
    
    # Read in the activity file and get the number of batch samples
    # Should be a list of length batch size, where each item in the list is an array 
    # of shape (input_width / stride x) x (input height / stride y) x n_neurons
    # with the activations for that particular batch sample
    act_data = octave.readpvpfile(model_activity_fpath)
    n_batch = len(act_data)
    
    # concatenate across batch samples to produce a single array of shape 
    # batch size x (input width / stride x) x (input height / stride y) x n_neurons
    acts = np.concatenate([act_data[b]['values'][None, ...] for b in range(n_batch)], 0)

    # get the mean activation for each neuron across batch and spatial dimensions
    # and sort them descending based on this mean and keep the neuron index too
    mean_acts = np.mean(acts, (0, 1, 2))
    inds = list(range(mean_acts.size))
    sorted_inds = [ind for _,ind in sorted(zip(mean_acts, inds), reverse = True)]
    sorted_acts = acts[..., sorted_inds]
    sorted_mean_acts = mean_acts[sorted_inds]
    
    return sorted_acts, sorted_mean_acts, sorted_inds


def view_complex_cell_strfs(ckpt_dir, write_fpath, weight_file_key = None, 
        activity_fpath = None, openpv_path = '/home/mteti/OpenPV/mlab/util'):
    '''
    View complex cell (spatially-shared weights) strfs in a grid.
    
    Args:
        ckpt_dir (str): The path to the Petavision checkpoint directory containing
            the *_W.pvp files with the weights in them.
        write_fpath (str): The file path to save the resulting .gif as. The parent
            directory should already exist.
        weight_file_key (str): A str that helps this function find and isolate the 
            weight files in the ckpt_dir without including other files (since there
            will be multiple weight files, one for each input video frame, and
            depending on your model, there might be multiple connection types that end
            in _W.pvp. Keeping this argument set to None will usually work for most 
            one-layer sparse coding models.
        activity_fpath (str): The path to the <model_layer_name>_A.pvp file in the 
            ckpt_dir. If this is given, the features will be sorted descending based
            on their mean activity spatially and across the batch samples. 
        openpv_path (str): Path to the openpv matlab utility directory (*/OpenPV/mlab/util).
        
    Returns:
        None
    '''
    
    # add OpenPV matlab utility dir to octave path
    octave.addpath(openpv_path)
    
    # get the paths to all of the weight files and sort ascending based on 
    # video frame number 
    if weight_file_key:
        weight_fpaths = glob(os.path.join(ckpt_dir, weight_file_key))
    else:
        weight_fpaths = glob(os.path.join(ckpt_dir, '*_W.pvp'))
        
    weight_fpaths.sort()
    n_frames = len(weight_fpaths) # number of video frames in the input
    
    # get descending indices each neuron's activation over the last batch 
    if activity_fpath:
        _, _, inds = get_mean_activations(activity_fpath, openpv_path = openpv_path)
        
    # loop through each weight file, extract the weights, and aggregate them
    for frame_num, weight_fpath in enumerate(weight_fpaths):
        # original shape: patch width x patch height x in channels x out channels
        # reshape to: out_channels x in channels x patch height x patch width
        weights = octave.readpvpfile(weight_fpath)[0]['values'][0]
        weights = weights.transpose([3, 2, 1, 0])
        n_neurons, in_c, patch_height, patch_width = weights.shape
        
        # sort the features based on the activation indices if given
        if activity_fpath:
            assert len(inds) == n_neurons
            weights = weights[inds]
        
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
    
    
def view_simple_cell_strfs(ckpt_dir, save_dir, n_features_y, n_features_x, 
        weight_file_key = None, openpv_path = '/home/mteti/OpenPV/mlab/util'):
    '''
    View simple cell (no spatial sharing of weights) strfs in a grid.
    
    Args:
        ckpt_dir (str): The path to the Petavision checkpoint directory containing
            the *_W.pvp files with the weights in them.
        save_dir (str): The directory to save the feature grids in, since there will
            be multiple grids.
        n_features_y (int): The number of features stacked vertically per grid.
        n_features_x (int): The number of features stacked horizontally per grid.
        weight_file_key (str): weight_file_key (str): A str that helps this function find and isolate the 
            weight files in the ckpt_dir without including other files (since there
            will be multiple weight files, one for each input video frame, and
            depending on your model, there might be multiple connection types that end
            in _W.pvp. Keeping this argument set to None will usually work for most 
            one-layer sparse coding models.
        openpv_path (str): Path to the openpv matlab utility directory (*/OpenPV/mlab/util).
        
    Returns:
        None
    '''
    
    # make sure save_dir exists or create it
    os.makedirs(save_dir, exist_ok = True)

    # add OpenPV matlab utility directory to octave path
    octave.addpath(openpv_path)

    # get the weight files
    if weight_file_key:
        weight_fpaths = glob(os.path.join(ckpt_dir, weight_file_key))
    else:
        weight_fpaths = glob(os.path.join(ckpt_dir, '*_W.pvp'))

    weight_fpaths.sort()
    n_frames = len(weight_fpaths)

    for frame_num, weight_fpath in enumerate(weight_fpaths):
        # read in the features and get the shape of each dimension 
        weights = octave.readpvpfile(weight_fpath)[0]['values'][0]
        w_x, w_y, w_in, w_out = weights.shape
        n_feats = w_out // n_features_x // n_features_y
        
        # reshape to the original shape of the unshared weight tensor 
        # and reverse the x and y dims for image writing
        weights = weights.reshape([w_x, w_y, w_in, n_feats, n_features_x, n_features_y])
        weights = weights.transpose([1, 0, 2, 3, 5, 4]) 
        
        for feat_num in range(n_feats):
            # reshape to use make_grid due to python ordering
            weights_feat = weights[:, :, :, feat_num, :, :]
            weights_feat = weights_feat.reshape([w_y, w_x, w_in, -1])
            weights_feat = weights_feat.transpose([3, 2, 0, 1])
            
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
            if frame_num == 0 and feat_num == 0:
                grids = np.zeros([n_feats, n_frames] + list(grid.shape))
                
            grids[feat_num, frame_num] = grid
            
    # scale all values to [0, 255]
    grids = np.uint8(max_min_scale(grids) * 255)
    
    # add makeshift black border between feats
    grids[:, :, ::w_y, :, :] = 0.0
    grids[:, :, :, ::w_x, :] = 0.0
    
    # save the grids per feature
    for feat_num in range(n_feats):
        imageio.mimwrite(
           os.path.join(save_dir, 'feature{}.gif'.format(feat_num)),
           [grids[feat_num, frame_num] for frame_num in range(n_frames)]
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
    
    acts, _, _ = get_mean_activations(model_activity_fpath, openpv_path = openpv_path)
    mean_sparsity = np.mean(acts == 0.0, (0, 1, 2))
    inds = list(range(mean_sparsity.size))
    sorted_inds = [ind for _,ind in sorted(zip(mean_sparsity, inds), reverse = True)]
    sorted_mean_sparsity = mean_sparsity[sorted_inds]
    
    return sorted_mean_sparsity, sorted_inds


def get_num_neurons_active(sparse_activity_fpath, openpv_path = '/home/mteti/OpenPV/mlab/util'):
    '''
    Computes the mean number of neurons active over each batch. 
    
    Args:
        sparse_activity_fpath (str): The <model_layer>.pvp file that contains the activities in 
            sparse format (e.g. S1.pvp).
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
    means, ses = np.zeros([n_display_periods]), np.zeros([n_display_periods])
    
    for display_period in range(n_display_periods):
        acts_over_batch = acts[display_period * n_batch: (display_period + 1) * n_batch]
        n_active_over_batch = [sample_acts['values'].shape[0] for sample_acts in acts_over_batch]
        means[display_period] = np.mean(np.array(n_active_over_batch, dtype = np.float64))
        ses[display_period] = np.std(np.array(n_active_over_batch, dtype = np.float64)) / sqrt_n
        
    return means, ses