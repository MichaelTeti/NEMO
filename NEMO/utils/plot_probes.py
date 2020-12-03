'''
Script to read in the four main probe output files written by PetaVision. 
Each one has a different function because each probe's output text file 
is formatted differently.
'''

from argparse import ArgumentParser
from glob import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from NEMO.utils.general_utils import read_csv



def line_plot(datay, fpath, datax = None, xlabel = None, ylabel = None):
    plt.plot(datay) if datax is None else plt.plot(datax, datay)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.savefig(fpath)
    plt.close()
    
def errorbar_plot(data, errorbars, fpath, xlabel = None, ylabel = None):
    plt.errorbar(x = list(range(1, data.shape[0] + 1)), y = data, yerr = errorbars)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.savefig(fpath)
    plt.close()



def plot_l2(ckpt_dir, save_dir, key, display_period = 3000, n_display_periods = None):
    os.makedirs(save_dir, exist_ok = True)

    fpaths = glob(os.path.join(ckpt_dir, key))
    l2_min = []
    
    for fpath in fpaths:
        data = np.genfromtxt(
            fpath,
            usecols = (-1)
        )
        timesteps = np.arange(data.size)
        line_plot(
            data[-n_display_periods * display_period:] if n_display_periods else data,
            os.path.join(save_dir, os.path.split(fpath)[1].split('.')[0] + '.png'),
            datax = timesteps[-n_display_periods * display_period:] if n_display_periods else None,
            xlabel = 'Step',
            ylabel = 'L2 Probe'
        )
        
        l2_min.append(data[timesteps % display_period == 0])
        
    l2_min = np.array(l2_min, dtype = np.float32)
    l2_min_mean = np.mean(l2_min, 0)
    l2_min_se = np.std(l2_min, 0) / np.sqrt(l2_min.shape[0])
    errorbar_plot(
        l2_min_mean[1:],
        l2_min_se[1:], 
        os.path.join(save_dir, 'minimum_l2.png'),
        xlabel = 'Display Period',
        ylabel = 'Mean Ending L2 Probe Value / Batch +/- SE'
    )
    
    
    
def plot_energy(ckpt_dir, save_dir, key, display_period = 3000, n_display_periods = None):
    os.makedirs(save_dir, exist_ok = True)
    
    energy_fpaths = glob(os.path.join(ckpt_dir, key))
    energy_min = []

    for fpath in energy_fpaths:
        data = np.genfromtxt(
            fpath,
            delimiter = ',',
            skip_header = 1,
            usecols = (0, 2),
            skip_footer = 1
        )
        line_plot(
            data[-n_display_periods * display_period:, -1] if n_display_periods else data[:, -1],
            os.path.join(save_dir, os.path.split(fpath)[1].split('.')[0] + '.png'),
            datax = data[-n_display_periods * display_period:, 0] if n_display_periods else None,
            xlabel = 'Step',
            ylabel = 'Energy'
        )
        
        energy_min.append(data[data[:, 0] % display_period == 0, -1])
        
    energy_min = np.array(energy_min, dtype = np.float32)
    energy_min_mean = np.mean(energy_min, 0)
    energy_min_se = np.std(energy_min, 0) / np.sqrt(energy_min.shape[0])
    errorbar_plot(
        energy_min_mean,
        energy_min_se, 
        os.path.join(save_dir, 'minimum_energy.png'),
        xlabel = 'Display Period',
        ylabel = 'Mean Ending Energy Value / Batch +/- SE'
    )
        
        
def plot_firm_thresh(ckpt_dir, save_dir, key, display_period = 3000, n_display_periods = None):
    os.makedirs(save_dir, exist_ok = True)

    fpaths = glob(os.path.join(ckpt_dir, key))
    firm_thresh_min = []
    
    for fpath in fpaths:
        data = np.genfromtxt(
            fpath,
            delimiter = ',',
            usecols = (0, 3),
            skip_footer = 1
        )
        line_plot(
            data[-n_display_periods * display_period:, -1] if n_display_periods else data[:, -1],
            os.path.join(save_dir, os.path.split(fpath)[1].split('.')[0] + '.png'),
            datax = data[-n_display_periods * display_period:, 0] if n_display_periods else None,
            xlabel = 'Step',
            ylabel = 'L1-Norm'
        )
        
        firm_thresh_min.append(data[data[:, 0] % display_period == 0, -1])
        
    firm_thresh_min = np.array(firm_thresh_min, dtype = np.float32)
    firm_thresh_min_mean = np.mean(firm_thresh_min, 0)
    firm_thresh_min_se = np.std(firm_thresh_min, 0) / np.sqrt(firm_thresh_min.shape[0])
    errorbar_plot(
        firm_thresh_min_mean[1:],
        firm_thresh_min_se[1:],
        os.path.join(save_dir, 'minimum_firm_thresh.png'),
        xlabel = 'Display Period',
        ylabel = 'Mean Ending L1-Norm / Batch +/- SE'
    )


def plot_adaptive_timescales(ckpt_dir, save_dir, key, display_period = 3000, n_display_periods = None):
    os.makedirs(save_dir, exist_ok = True)

    timescales_paths = glob(os.path.join(ckpt_dir, key))

    for fpath in timescales_paths:
        data = read_csv(fpath)
        times = [float(row.split(' = ')[1]) for row in data[0::2]]
        ts = [float(row[1].split(' = ')[1]) for row in data[1::2]]
        ts_true = [float(row[2].split(' = ')[1]) for row in data[1::2]]
        ts_max = [float(row[3].split(' = ')[1]) for row in data[1::2]]

        save_fname = os.path.join(save_dir, os.path.split(fpath)[1].split('.')[0] + '_')
        line_plot(
            ts[-n_display_periods * display_period:] if n_display_periods else ts,
            save_fname + 'timescale.png',
            datax = times[-n_display_periods * display_period:] if n_display_periods else None,
            xlabel = 'Step',
            ylabel = 'Timescale'
        )
        line_plot(
            ts_true[-n_display_periods * display_period:] if n_display_periods else ts_true,
            save_fname + 'timescale_true.png',
            datax = times[-n_display_periods * display_period:] if n_display_periods else None,
            xlabel = 'Step',
            ylabel = 'Timescale True'
        )
        line_plot(
            ts_max[-n_display_periods * display_period:] if n_display_periods else ts_max,
            save_fname + 'timescale_max.png',
            datax = times[-n_display_periods * display_period:] if n_display_periods else None,
            xlabel = 'Step',
            ylabel = 'Timescale Max'
        )

        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'ckpt_dir',
        type = str,
        help = 'The directory with the .txt files to plot.'
    )
    parser.add_argument(
        'save_dir',
        type = str,
        help = 'The directory to save the plots in.'
    )
    parser.add_argument(
        'key',
        type = str,
        help = 'The key to filter out the .txt files you want.'
    )
    parser.add_argument(
        'plot_type',
        choices = ['energy', 'adaptivetimescales', 'firmthresh', 'l2'],
        type = str
    )
    parser.add_argument(
        '--display_period',
        type = int,
        default = 3000,
        help = 'The length of the display period.'
    )
    parser.add_argument(
        '--n_display_periods',
        type = int,
        help = 'Number of display periods to display starting from the end and moving backwards. Defaults to all.'
    )
    args = parser.parse_args()

    assert(os.path.isdir(args.ckpt_dir)), \
        "Given ckpt_dir doesn't exist."
    
    os.makedirs(args.save_dir, exist_ok = True)

    if args.plot_type == 'energy':
        plot_energy(
            args.ckpt_dir, 
            args.save_dir, 
            args.key, 
            display_period = args.display_period,
            n_display_periods = args.n_display_periods
        )
    elif args.plot_type == 'adaptivetimescales':
        plot_adaptive_timescales(
            args.ckpt_dir, 
            args.save_dir, 
            args.key,
            display_period = args.display_period,
            n_display_periods = args.n_display_periods
        )
    elif args.plot_type == 'firmthresh':
        plot_firm_thresh(
            args.ckpt_dir, 
            args.save_dir, 
            args.key,
            display_period = args.display_period,
            n_display_periods = args.n_display_periods
        )
    elif args.plot_type == 'l2':
        plot_l2(
            args.ckpt_dir,
            args.save_dir,
            args.key,
            display_period = args.display_period,
            n_display_periods = args.n_display_periods
        )
