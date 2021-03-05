import os

from allensdk.brain_observatory.stimulus_info import BrainObservatoryMonitor
import cv2
import numpy as np

from nemo.data.preprocess.image import max_min_scale
from nemo.data.utils import get_img_frame_names


def write_vid_frames(vid_array, save_dir, scale_method = None):
    '''
    Save a video represented as an array as individual image files.

    Args:
        array (np.ndarray) Array of shape B x H x W x C or B x H x W to write.
        dir (str): Directory to write the frames in.
        scale_method (None, "video", "frame"): If none, no pixel value scaling will be
            performed. Otherwise, "video" will scale every frame by the video's max and 
            min, and "frame" will scale every frame by the frame's max and min.

    Returns:
        None
    '''

    os.makedirs(save_dir, exist_ok = True)
    n_frames = vid_array.shape[0]
    fpaths = [os.path.join(save_dir, fname + '.png') for fname in get_img_frame_names(n_frames)]
    
    if scale_method and scale_method == 'video':
        vid_array = max_min_scale(vid_array) * 255
    
    for i_frame, (frame, fpath) in enumerate(zip(vid_array, fpaths)):
        if scale_method and scale_method == 'frame':
            frame = max_min_scale(frame) * 255

        cv2.imwrite(fpath, np.uint8(frame))


def read_frames(dir, return_type = 'array', gray = False):
    '''
    Traverse a directory structure, reading in all images along the way and returning as list or np.ndarray.

    Args:
        dir (str): Directory to start at.
        return_type ('array' or 'list'): Data structure to return the video frames as.
        gray (bool): If true, return grayscale frames.

    Returns:
        Video frames.
    '''

    frames = []
    for root, dirs, files in os.walk(dir):
        files.sort()

        for file in files:

            if os.path.splitext(file)[1] in ['.jpeg', '.jpg', '.JPG', '.JPEG', '.PNG', '.png']:
                frame = cv2.imread(os.path.join(root, file))
                
                if gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
                frames.append(frame)

    if return_type == 'array':
        return np.array(frames)
    elif return_type == 'list':
        return frames


def write_AIBO_natural_stimuli(template, save_dir, stimulus):
    '''
    Takes the natural_movie_* or natural_scenes stimulus template, and 
    writes the images/frames as they would appear on the monitor. 
    '''

    monitor = BrainObservatoryMonitor()

    os.makedirs(save_dir, exist_ok = True)
    fnames = [fname + '.png' for fname in get_img_frame_names(template.shape[0])]

    for image, fname in zip(template, fnames):
        if 'natural_movie' in stimulus:
            image = monitor.natural_movie_image_to_screen(image, origin = 'upper')
        elif stimulus == 'natural_scenes':
            image = monitor.natural_scene_image_to_screen(image, origin = 'upper')

        cv2.imwrite(
            os.path.join(save_dir, fname), 
            monitor.warp_image(image)
        )


def write_AIBO_static_grating_stimuli(stim_table, save_dir):
    '''
    Obtains and writes the static grating stimuli from the AIBO database.
    '''

    monitor = BrainObservatoryMonitor()
    os.makedirs(save_dir, exist_ok = True)
    
    for orient in stim_table['orientation'].unique():
        for freq in stim_table['spatial_frequency'].unique():
            for phase in stim_table['phase'].unique():
                if np.isnan(orient) or np.isnan(freq) or np.isnan(phase):
                    continue
                    
                fname = '{}_{}_{}.png'.format(orient, freq, phase)
                if fname not in os.listdir(save_dir):
                    cv2.imwrite(
                        os.path.join(save_dir, fname), 
                        monitor.warp_image(
                            monitor.grating_to_screen(
                                phase = phase, 
                                spatial_frequency = freq, 
                                orientation = orient
                            )
                        )
                    )