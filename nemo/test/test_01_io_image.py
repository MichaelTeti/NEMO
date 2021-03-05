''' 
Unit tests for the functions in data.io.image.py
'''


import os
from tempfile import TemporaryDirectory
import unittest 

import cv2
import numpy as np
import pandas as pd

from nemo.data.io.image import (
    read_frames,
    save_vid_array,
    write_AIBO_natural_stimuli,
    write_AIBO_static_grating_stimuli
)


class TestIOImage(unittest.TestCase):

    def test_save_vid_array(self):
        write = np.random.uniform(0, 255, size = (10, 32, 64))
        write = np.uint8(write)

        with TemporaryDirectory() as tmp_dir:
            save_vid_array(write, tmp_dir)
            files = os.listdir(tmp_dir)
            files.sort()
            read = np.zeros([10, 32, 64])

            for i, fname in enumerate(files):
                read[i] = cv2.imread(os.path.join(tmp_dir, fname), cv2.IMREAD_GRAYSCALE)

            self.assertEqual(np.sum(read - write), 0.0)


    def test_read_frames(self):
        write = np.random.uniform(0, 255, size = (10, 32, 64))
        write = np.uint8(write)

        with TemporaryDirectory() as tmp_dir:
            for i in range(10):
                cv2.imwrite(os.path.join(tmp_dir, '0{}.png'.format(i)), write[i])
            
            read = read_frames(tmp_dir, gray = True)
            self.assertEqual(np.sum(read - write), 0.0)


    def test_write_AIBO_natural_stimuli_natural_movie_stimulus_n_written(self):
        template = np.uint8(np.random.rand(10, 304, 608) * 255)
        
        with TemporaryDirectory() as tmp_dir:
            write_AIBO_natural_stimuli(
                template,
                tmp_dir, 
                'natural_movie_one'
            )
            self.assertEqual(len(os.listdir(tmp_dir)), 10)


    def test_write_AIBO_natural_stimuli_natural_movie_stimulus_shape(self):
        template = np.uint8(np.random.rand(10, 304, 608) * 255)
        
        with TemporaryDirectory() as tmp_dir:
            write_AIBO_natural_stimuli(
                template,
                tmp_dir, 
                'natural_movie_one'
            )
            written = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
            
            for fpath in written:
                image = cv2.imread(fpath) 
                self.assertCountEqual(image.shape[:2], [1200, 1920])


    def test_write_AIBO_natural_stimuli_natural_scene_stimulus_shape(self):
        template = np.uint8(np.random.rand(10, 918, 1174) * 255)
        
        with TemporaryDirectory() as tmp_dir:
            write_AIBO_natural_stimuli(
                template,
                tmp_dir, 
                'natural_scenes'
            )
            written = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
            
            for fpath in written:
                image = cv2.imread(fpath) 
                self.assertCountEqual(image.shape[:2], [1200, 1920])


    def test_write_AIBO_natural_stimuli_natural_scene_stimulus_n_written(self):
        template = np.uint8(np.random.rand(10, 918, 1174) * 255)
        
        with TemporaryDirectory() as tmp_dir:
            write_AIBO_natural_stimuli(
                template,
                tmp_dir, 
                'natural_scenes'
            )
            self.assertEqual(len(os.listdir(tmp_dir)), 10)


    def test_write_AIBO_static_grating_stimuli_n_written(self):
        stim_table = pd.DataFrame(
            {
                'orientation': [0, 45, 90, 90],
                'spatial_frequency': [0.01, 0.05, 0.1, 0.1],
                'phase': [0.0, 0.25, 0.5, 0.25]
            }
        )
        
        with TemporaryDirectory() as tmp_dir:
            write_AIBO_static_grating_stimuli(stim_table, tmp_dir)
            self.assertEqual(len(os.listdir(tmp_dir)), 3 ** 3)


    def test_write_AIBO_static_grating_stimuli_shape(self):
        stim_table = pd.DataFrame(
            {
                'orientation': [0, 45, 90, 135],
                'spatial_frequency': [0.01, 0.05, 0.1, 0.15],
                'phase': [0.0, 0.25, 0.5, 0.75]
            }
        )
        
        with TemporaryDirectory() as tmp_dir:
            write_AIBO_static_grating_stimuli(stim_table, tmp_dir)
            written = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]

            for fpath in written:
                image = cv2.imread(fpath)
                self.assertCountEqual(image.shape[:2], [1200, 1920])



if __name__ == '__main__':
    unittest.main(verbosity = 2)