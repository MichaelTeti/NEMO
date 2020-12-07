from argparse import ArgumentParser
from glob import glob
import os

import h5py
import numpy as np
from oct2py import octave
import pandas as pd

parser = ArgumentParser()
parser.add_argument(
    'lca_rf_dir',
    type = str,
    help = 'The directory containing the *_W.pvp files with the learned LCA features.'
)
parser.add_argument(
    'mouse_rf_fpath',
    type = str,
    help = 'The path to the file rfs.h5 file with the strfs for different \
        cells in the allen institute database.'
)
parser.add_argument(
    'save_dir',
    type = str,
    help = 'The directory to save the modified *_W.pvp files with the mouse rfs \
        added in.'
)
parser.add_argument(
    'openpv_path',
    type = str,
    help = 'The path to the OpenPV/mlab/util directory.'
)
args = parser.parse_args()

# make sure the save_dir exists
os.makedirs(args.save_dir, exist_ok = True)

# add openpv matlab utility path to read .pvp files
octave.addpath(args.openpv_path)

# get a list of the *_W.pvp files in lca_rf_dir
lca_fpaths = glob(os.path.join(args.lca_rf_dir, '*_W.pvp'))
lca_fpaths.sort()

# read in the mouse strfs from the .h5 file
with h5py.File(args.mouse_rf_fpath, 'r') as h5file:
    cell_ids = list(h5file.keys())
    n_cells = len(cell_ids)
    cell_inds = []

    for cell_num, cell_id in enumerate(cell_ids):
        strf = h5file[cell_id][()]
        cell_inds.append(cell_num)
        
        if cell_num == 0:
            h, w, n_frames = strf.shape
            mouse_strfs = np.zeros([w, h, n_frames, n_cells], dtype = np.float64)
            
        mouse_strfs[..., cell_num] = strf.transpose([1, 0, 2])


# loop through each individual .pvp file
for frame_num, lca_fpath in enumerate(lca_fpaths):
    # read in the pvp files and get the lca features
    pvp_data = octave.readpvpfile(lca_fpath)
    weights = pvp_data[0]['values'][0]
    w_out_original = weights.shape[-1]    

    # take out the current frame from the strf and add to the weights
    mouse_strfs_frame = mouse_strfs[:, :, frame_num, :]
    mouse_strfs_frame = mouse_strfs_frame[:, :, None, :] 
    weights = np.concatenate((weights, mouse_strfs_frame), 3)
    
    # add the new weights back to the pvp_data structure and push to octave session
    pvp_data[0]['values'][0] = weights
    write_fpath = os.path.join(args.save_dir, os.path.split(lca_fpath)[1])
    octave.push(['pvp_data', 'write_fpath'], [pvp_data, write_fpath])
    
    # write the new pvp weight file
    octave.eval('writepvpsharedweightfile(write_fpath, pvp_data)')
    
print('[INFO] THE NEW NUMBER OF FEATURES IS {}.'.format(weights.shape[-1]))    

# write out the indices for each cell in the dictionary
cell_ids = [cell_id.split('_')[1] for cell_id in cell_ids]
cell_inds = [cell_ind + w_out_original for cell_ind in cell_inds]
df = pd.DataFrame(zip(cell_ids, cell_inds), columns = ['CellID', 'FeatureIndex'])
df.to_csv(os.path.join(args.save_dir, 'cell_inds.txt'), index = False)
