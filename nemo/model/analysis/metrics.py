import numpy as np


def lifetime_sparsity(feat_maps, eps = 1e-12):
    '''
    Computes lifetime sparsity for each neuron as in 
    https://www.biorxiv.org/content/biorxiv/suppl/2018/06/29/359513.DC1/359513-1.pdf.

    Args:
        feat_maps (np.ndarray): The feature maps of shape B x H x W x # Neurons. 
            If H and W are equal to 1, then the lifetime sparsity is computed over
            the batch. Otherwise, it is computed over the batch and spatial dims.

    Returns lifetime_sparsity (np.ndarray): A vector of shape # Neurons with the 
        lifetime sparsity of each neuron. 
    '''

    n_neurons = feat_maps.shape[-1]
    feat_maps = feat_maps.reshape([-1, n_neurons])
    B = feat_maps.shape[0]
    
    sum_square = np.sum(feat_maps, 0) ** 2
    square_sum = np.sum(feat_maps ** 2, 0)

    return (1 - (1 / B) * sum_square / (square_sum + eps)) / (1 - 1 / B)