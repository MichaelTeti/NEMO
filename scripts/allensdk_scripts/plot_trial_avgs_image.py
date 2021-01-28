from argparse import ArgumentParser
import os, sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn

from nemo.data.io.trace import compile_trial_avg_traces


parser = ArgumentParser()
parser.add_argument(
    'trace_dir',
    type = str,
    help = 'Directory where the fluorescence trace .txt files are located.'
)
parser.add_argument(
    'save_fpath',
    type = str,
    help = 'Directory where the trace images will be saved.'
)
parser.add_argument(
    '--plot_title',
    type = str,
    help = 'Title of the plot.'
)
args = parser.parse_args()


os.makedirs(os.path.split(args.save_fpath)[0], exist_ok = True)

# get traces and cell_ids
df, cell_ids = compile_trial_avg_traces(args.trace_dir)

# plot the image and save
df.columns = list(range(len(df.columns)))
seaborn.heatmap(data = df, cmap = 'viridis')
plt.xlabel('Frame Number')
plt.ylabel('Neuron Number')
if args.plot_title: plt.title(args.plot_title)
plt.savefig(args.save_fpath, bbox_inches = 'tight', dpi = 300)
