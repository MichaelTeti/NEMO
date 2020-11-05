import os, sys
sys.path.append('/'.join(os.getcwd().split('/')[:-3]))
from v1_response_prediction.utils.general_utils import multiproc, read_csv, write_csv

from argparse import ArgumentParser
from imageio import imread, imwrite
import numpy as np
from fancyimpute import SoftImpute, IterativeSVD, MatrixFactorization, BiScaler
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt


parser = ArgumentParser()
parser.add_argument('data_dir_parent',
    type = str,
    help = 'Directory to recurse through to find csv files with the data.')
parser.add_argument('--key',
    type = str,
    help = 'A key phrase used to filter out desired folders. Default is None.')
args = parser.parse_args()

# where were going to save the interpolated data files
save_dir = args.data_dir_parent + '_interp' if args.data_dir_parent[-1] != '/' else args.data_dir_parent[:-1] + '_interp'
os.makedirs(save_dir, exist_ok = True)

# interpolate one stimulus at a time to match the number of coordinates
fnames = os.listdir(args.data_dir_parent)
stimuli = list(set([fname.split('_')[-2] for fname in fnames]))

for stimulus in stimuli:
    fnames_stimulus = [fname for fname in fnames if stimulus in fname]
    fpaths_stimulus = [os.path.join(args.data_dir_parent, fname) for fname in fnames_stimulus]
    scale_vals = []

    for i_fpath, fpath in enumerate(fpaths_stimulus):
        data = read_csv(fpath) # read in the data with missing values

        # put zeros in the place of the missing values
        for i_coords, coords in enumerate(data):
            if 'nan' in coords:
                data[i_coords] = [0.0, 0.0]
            else:
                data[i_coords] = [float(coord) + 1e-6 for coord in data[i_coords]]

        # flatten and scale the data and then add the nans back in
        data = np.array(data).flatten()
        max_abs = np.amax(np.absolute(data))
        data /= max_abs
        scale_vals.append(max_abs)
        data[data == 0] = np.nan
        if i_fpath == 0:
            data_agg = np.zeros([len(fpaths_stimulus), data.size])

        data_agg[i_fpath, :] = data # aggregate the data into one array

    # interpolation algorithm
    data_interp = SoftImpute().fit_transform(data_agg)

    # for each file, scale it back and save it
    for i_fpath, fpath in enumerate(fpaths_stimulus):
        interp_sample = data_interp[i_fpath].reshape([-1, 2]) * scale_vals[i_fpath]
        interp_sample_list = [list(coords) for coords in interp_sample]
        save_fpath = os.path.join(save_dir, os.path.split(fpath)[1])
        write_csv(interp_sample_list, save_fpath)

    plt.plot(data_agg[0][::2], c = 'r')
    plt.savefig('test_orig_x_{}.png'.format(stimulus))
    plt.close()
    plt.plot(data_interp[0][::2], c = 'b')
    plt.savefig('test_interp_x_{}.png'.format(stimulus))
    plt.close()


    plt.plot(data_agg[0][1::2], c  = 'r')
    plt.savefig('test_orig_y_{}.png'.format(stimulus))
    plt.close()
    plt.plot(data_interp[0][1::2], c = 'b')
    plt.savefig('test_interp_y_{}.png'.format(stimulus))
    plt.close()
