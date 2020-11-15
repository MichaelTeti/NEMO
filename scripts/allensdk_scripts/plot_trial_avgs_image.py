from argparse import ArgumentParser
import os, sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn


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

assert(os.path.isdir(args.trace_dir) and os.listdir(args.trace_dir) != []), \
    'trace_dir must exist and must not be empty.'
assert(os.path.splitext(args.save_fpath)[1] in ['.png', '.jpg', '.PNG', '.jpeg', '.JPG']), \
    'save_fpath must have an image extension.'

os.makedirs(os.path.split(args.save_fpath)[0], exist_ok = True)

# get paths to all the trace files in the given trace_dir
fnames = os.listdir(args.trace_dir)
fnames.sort() # sort them so they'll be lined up regardless of stimuli etc.
fpaths = [os.path.join(args.trace_dir, f) for f in fnames]

# loop through and read them all into a pandas dataframe
for fpath_num, fpath in enumerate(fpaths):
    if fpath_num == 0:
        df = pd.read_csv(fpath)
    else:
        df = df.append(pd.read_csv(fpath))

# plot the image and save
df = df.reset_index(drop = True)
df.columns = list(range(len(df.columns)))
seaborn.heatmap(data = df, cmap = 'viridis')
plt.xlabel('Frame Number')
plt.ylabel('Neuron Number')
if args.plot_title: plt.title(args.plot_title)
plt.savefig(args.save_fpath, bbox_inches = 'tight', dpi = 300)
