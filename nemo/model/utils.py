import json
import os
from random import shuffle

import numpy as np
from pytorch_lightning.trainer.trainer import Trainer as PLTrainer
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


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


def tune_model(config, ptl_model, dset, train_inds, n_workers, n_val, tune_metrics = None, 
               mode = 'tune', **trainer_kwargs):
    ''' A generic function to hp-tuning and model training with ray and pytorch-lightning '''
    
    model = ptl_model(config = config)
    
    shuffle(train_inds)
    train_dl = DataLoader(
        dset,
        batch_size = config['batch_size'],
        num_workers = n_workers,
        sampler = RandomSampler(train_inds[n_val:]),
        drop_last = True
    )
    val_dl = DataLoader(
        dset,
        num_workers = n_workers,
        batch_size = config['batch_size'],
        sampler = SequentialSampler(train_inds[:n_val]),
        drop_last = True
    )
    
    callbacks = model.callbacks
    if mode == 'tune':
        callbacks += [
            TuneReportCallback(
                tune_metrics, 
                on = 'validation_end'
            )
        ]

    trainer = PLTrainer(callbacks = callbacks, **trainer_kwargs)
    trainer.fit(model, train_dl, val_dl)
    
    return trainer