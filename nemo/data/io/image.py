import os

from allensdk.brain_observatory.stimulus_info import BrainObservatoryMonitor
import cv2
import imageio
import numpy as np

from nemo.data.preprocess.image import max_min_scale
from nemo.data.utils import get_img_frame_names


def write_vid_frames(vid_array, save_dir, scale_method = None):
    '''
    Write a video represented as an array as individual image files.

    Args:
        array (np.ndarray) Array of shape N x H x W x C or N x H x W to write.
        dir (str): Directory to write the frames in.
        scale_method (None, "video", "frame"): If none, no pixel value scaling will be
            performed. Otherwise, "video" will scale every frame by the video's max and 
            min, and "frame" will scale every frame by the frame's max and min.

    Returns:
        None
    '''

    os.makedirs(save_dir, exist_ok = True)
    
    if scale_method and scale_method == 'video':
        vid_array = max_min_scale(vid_array) * 255
    
    for i_frame, frame in enumerate(vid_array):
        if scale_method and scale_method == 'frame':
            frame = max_min_scale(frame) * 255

        cv2.imwrite(
            os.path.join(save_dir, '{}.png'.format(i_frame)), 
            np.uint8(frame)
        )


def write_gifs(array, save_dir, scale = False):
    '''
    Write a batch of video frame sequences as .gifs.

    Args:
        array (np.ndarray) Array of shape N x F x H x W x C or N x F x H x W to write, 
            where F is the number of consecutive frames to write in each gif.
        dir (str): Directory to write the frames in.
        scale (bool): If True, will scale each gif linearly to [0, 255].

    Returns:
        None
    '''

    os.makedirs(save_dir, exist_ok = True)

    for i_gif, gif in enumerate(array):
        if scale:
            gif = max_min_scale(gif) * 255 
        
        imageio.mimwrite(
            os.path.join(save_dir, '{}.gif'.format(i_gif)),
            [np.uint8(frame) for frame in gif]
        )


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


def write_AIBO_natural_stimuli(template, save_dir, stimulus, height = 160, width = 256):
    '''
    Takes the natural_movie_* or natural_scenes stimulus template, and 
    writes the images/frames as they would appear on the monitor. 
    '''

    monitor = BrainObservatoryMonitor()

    os.makedirs(save_dir, exist_ok = True)
    fnames = [fname + '.png' for fname in get_img_frame_names(template.shape[0])]
    
    # scale to [0, 255]
    template = np.uint8(max_min_scale(template) * 255)

    for image, fname in zip(template, fnames):

        # try to filter out some of the pixelation
        image = cv2.bilateralFilter(image, 7, 40, 40)

        if 'natural_movie' in stimulus:
            image = monitor.natural_movie_image_to_screen(image, origin = 'upper')
        elif stimulus == 'natural_scenes':
            image = monitor.natural_scene_image_to_screen(image, origin = 'upper')

        # warp image as it was shown on monitor
        image = monitor.warp_image(image)

        # resize
        image = cv2.resize(image, (width, height))

        # contrast enhance
        image = cv2.equalizeHist(image)

        cv2.imwrite(os.path.join(save_dir, fname), image)


def write_AIBO_static_grating_stimuli(stim_table, save_dir, height = 160, width = 256):
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
                    frame = monitor.warp_image(
                        monitor.grating_to_screen(
                            phase = phase, 
                            spatial_frequency = freq, 
                            orientation = orient
                        )
                    )
                    cv2.imwrite(
                        os.path.join(save_dir, fname), 
                        cv2.resize(frame, (width, height))
                    )