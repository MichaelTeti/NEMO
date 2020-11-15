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
    'save_dir',
    type = str,
    help = 'Directory where the trace images will be saved.'
)
parser.add_argument(
    '--ylabel',
    default = 'Mean Normalized Fluorescence Traces (95% CI)',
    type = str,
    help = 'y-axis label on plots.'
)
args = parser.parse_args()

assert(os.path.isdir(args.trace_dir) and os.listdir(args.trace_dir) != []), \
    'args.trace_dir does not exist or it is empty.'

os.makedirs(args.save_dir, exist_ok = True)

fpaths = [os.path.join(args.trace_dir, f) for f in os.listdir(args.trace_dir)]

for fpath in fpaths:
    df = pd.read_csv(fpath)
    cell_id = df['cell_id'].tolist()[0]
    df = df.filter(regex = '.png', axis = 1)
    df = df.melt(var_name = 'img_fname', value_name = 'response')
    df['frame_num'] = df['img_fname'].apply(lambda x: int(os.path.splitext(x)[0]))
    seaborn.lineplot(
        data = df,
        x = 'frame_num',
        y = 'response'
    )
    plt.ylabel(args.ylabel)
    plt.xlabel('Frame Number')
    plt.savefig(
        os.path.join(args.save_dir, 'cellID_{}.png'.format(cell_id)),
        bbox_inches = 'tight',
        dpi = 300
    )
    plt.close()
