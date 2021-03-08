import os

import imageio
import numpy as np
import torch
from torchvision.utils import make_grid

from nemo.data.preprocess.image import max_min_scale


def write_complex_cell_strfs(weight_tensors, write_fpath, sort_inds = None):
    '''
    View complex cell (spatially-shared weights) strfs in a grid.
    
    Args:
        weight_tensors (list): A list of np.ndarrays of shape out_c x in_c x kh x kw.
        write_fpath (str): The file path to save the resulting .gif as. 
        sort_inds (list): List of indices of length out_c to sort the features by in the grid.
        openpv_path (str): Path to the openpv matlab utility directory (*/OpenPV/mlab/util).
        
    Returns:
        None
    '''

    if os.path.split(write_fpath)[0] != '': 
        os.makedirs(os.path.split(write_fpath)[0], exist_ok = True)
    
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