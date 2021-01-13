'''
Script to make the .txt files that PetaVision will use to read in the input
images in order. 
'''

from argparse import ArgumentParser
import os, sys

from nemo.data.io import write_csv


parser = ArgumentParser()
parser.add_argument(
    'data_dir_parent',
    type = str,
    help = 'Directory to be searched recursively for training images.'
)
parser.add_argument(
    'save_dir',
    type = str,
    help = 'Dir to save the files in.'
)
parser.add_argument(
    '--key',
    type = str,
    help = 'Key to filter out certain folders when searching for images.'
)
parser.add_argument(
    '--n_frames_in_time',
    type = int,
    default = 3,
    help = 'The number of frames to model in time. Default is 3.'
)
parser.add_argument(
    '--n_videos',
    type = int,
    help = 'Number of videos to stop at.'
)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok = True)

vid_count = 0
for root, dirs, files in os.walk(args.data_dir_parent):
    if args.n_videos and vid_count == args.n_videos: break
    if args.key and args.key not in root: continue
    if len(files) < args.n_frames_in_time: continue
    files.sort()
    files = [os.path.join(root, f) for f in files]
    vid_count += 1

    for i in range(args.n_frames_in_time):
        img_paths_i = files[i:len(files) - (args.n_frames_in_time - i - 1)]
        save_path = os.path.join(args.save_dir, 'filenames_frame{}.txt'.format(i))
        write_csv(img_paths_i, save_path, mode = 'a')
