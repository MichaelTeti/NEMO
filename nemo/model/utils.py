import json
import os

import numpy as np
import torch


def cv_splitter_video(n_samples, n_folds = 5):
    '''
    Get cross validation train and validation indices.

    Args:
        n_samples (int): The number of samples in the dataset.
        n_folds (int): The number of cv folds. 

    Returns:
        fold_inds (list): A list of n_folds tuples of train inds and val inds.
    '''

    if n_folds <= 0:
        raise ValueError('n_folds should be > 0.')
    
    n_val = int(n_samples * (1 / n_folds))
    n_train = n_samples - n_val
    fold_inds = []

    for fold in range(n_folds):
        train_inds = np.arange(n_samples)
        val_inds = train_inds[fold * n_val:(fold + 1) * n_val]
        train_inds = np.delete(train_inds, val_inds)
        fold_inds.append((train_inds, val_inds))

    return fold_inds


def save_args(args, save_dir):
    '''
    Saves argparse args to a .txt file in save_dir.

    Args:
        args (namespace): The namespace returned by ArgumentParser.parse_args().
        save_dir (str): The directory where the args will be saved as args.txt.

    Returns:
        None
    '''

    arg_dict = vars(args)

    with open(os.path.join(save_dir, 'args.txt'), 'w') as fp:
        json.dump(arg_dict, fp, sort_keys = True, indent = 4)


def shuffle_design_mat(preds, responses):
    '''
    Shuffles the rows of the design matrix and the responses. 

    Args:
        preds (np.ndarray): The N x M dimensional design matrix with N samples and M predictors.
        responses (np.ndarray): The N-dimensional response vector corresponding to preds. 

    Returns:
        preds_shuffled (np.ndarray): The N x M dimensional design matrix with permuted rows. 
        responses_shuffled (np.ndarray): The N-dimensional response vector with permuted values matching 
            preds_shuffled. 
    '''

    data = np.concatenate((preds, responses[:, None]), 1)   
    np.random.shuffle(data)

    return data[:, :-1], data[:, -1]


def to_tensor(data, dev = None):
    ''' Makes data into torch tensor '''
    
    data = torch.FloatTensor(data)
    
    if dev is not None:
        data = data.cuda(dev)

    return data