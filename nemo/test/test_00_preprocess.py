''' 
Unit tests for the classes in preprocess.py
'''


import unittest 

import numpy as np

from nemo.data.preprocess import (
    create_video_frame_sequences,
    max_min_scale,
    normalize_traces,
    standardize_preds
)


class TestPreprocess(unittest.TestCase):

    def test_max_min_scale_values(self):
        rand_vec = np.random.randn(10000)
        rand_vec_scaled = max_min_scale(rand_vec)
        scaled_min = np.amin(rand_vec_scaled)
        scaled_max = np.amax(rand_vec_scaled)
        self.assertAlmostEqual(scaled_min, 0.0, places = 6)
        self.assertAlmostEqual(scaled_max, 1.0, places = 6)

    def test_max_min_scale_shape(self):
        rand_vec = np.random.randn(10000)
        rand_vec_scaled = max_min_scale(rand_vec)
        self.assertCountEqual(rand_vec_scaled.shape, rand_vec.shape)

    def test_create_video_frame_sequences_values(self):
        test_array = np.zeros([7, 32, 64])
        for i in range(7):
            test_array[i] = i
        
        test_sequences = create_video_frame_sequences(test_array, n_frames_in_time = 3)
        for i in range(5):
            self.assertEqual(np.sum(test_sequences[i, :, :, 0] - test_array[i]), 0)
            self.assertEqual(np.sum(test_sequences[i, :, :, 1] - test_array[i + 1]), 0)
            self.assertEqual(np.sum(test_sequences[i, :, :, 2] - test_array[i + 2]), 0)

    def test_create_video_frame_sequences_shape(self):
        test_array = np.zeros([7, 32, 64])
        test_sequences = create_video_frame_sequences(test_array, n_frames_in_time = 3)
        self.assertCountEqual([5, 32, 64, 3], test_sequences.shape)
            
    def test_normalize_traces_values(self):
        test_traces = np.random.randn(10000)
        test_traces_norm = normalize_traces(test_traces)
        norm_mean = np.mean(test_traces_norm)
        norm_max = np.amax(np.absolute(test_traces_norm))
        self.assertAlmostEqual(norm_mean, 0.0, places = 6)
        self.assertAlmostEqual(norm_max, 1.0, places = 6)

    def test_normalize_traces_shape(self):
        test_traces = np.random.randn(10000)
        test_traces_norm = normalize_traces(test_traces)
        self.assertCountEqual(test_traces_norm.shape, test_traces.shape)

    def test_standardize_preds_values(self):
        test_array = np.random.rand(1000, 32, 64)
        test_array_standardized = standardize_preds(test_array)
        mean_diff = np.sum(np.mean(test_array_standardized, 0) - np.zeros([32, 64]))
        std_diff = np.sum(np.std(test_array_standardized, 0) - np.ones([32, 64]))
        self.assertAlmostEqual(mean_diff, 0.0, places = 6)
        self.assertAlmostEqual(std_diff, 0.0, places = 6)

    def test_standardize_preds_shape(self):
        test_array = np.random.randn(1000, 32, 64)
        test_array_standardized = standardize_preds(test_array)
        self.assertCountEqual(test_array_standardized.shape, test_array.shape)


if __name__ == '__main__':
    unittest.main(verbosity = 2)