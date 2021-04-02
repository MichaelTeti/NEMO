''' 
Unit tests for the functions in model.analysis.metrics.py.
'''

import os
from tempfile import TemporaryDirectory
import unittest 

import numpy as np
import pandas as pd

from nemo.model.analysis.metrics import (
    lifetime_sparsity,
    population_sparsity,
    signal_power
)


class TestModelAnalysisMetrics(unittest.TestCase):

    def test_lifetime_sparsity_shape_hw_not_equal_1(self):
        feat_maps = np.zeros([10, 3, 5, 7])
        sparsity = lifetime_sparsity(feat_maps)
        self.assertEqual(sparsity.size, 7)


    def test_population_sparsity_shape_hw_not_equal_1(self):
        feat_maps = np.zeros([10, 3, 5, 7])
        sparsity = population_sparsity(feat_maps)
        self.assertEqual(sparsity.size, 10)


    def test_lifetime_sparsity_shape_hw_equal_1(self):
        feat_maps = np.zeros([10, 1, 1, 7])
        sparsity = lifetime_sparsity(feat_maps)
        self.assertEqual(sparsity.size, 7)


    def test_population_sparsity_shape_hw_equal_1(self):
        feat_maps = np.zeros([10, 1, 1, 7])
        sparsity = population_sparsity(feat_maps)
        self.assertEqual(sparsity.size, 10)


    def test_lifetime_sparsity_values(self):
        feat_maps = np.zeros([10, 1, 1, 3])
        feat_maps[:5, :, :, 1] = 1.0
        feat_maps[..., 2] = 1.0

        sparsity = lifetime_sparsity(feat_maps)
        self.assertAlmostEqual(sparsity[0], 1.0)
        self.assertAlmostEqual(sparsity[1], 0.55, places = 1)
        self.assertAlmostEqual(sparsity[2], 0.0)


    def test_population_sparsity_values(self):
        feat_maps = np.zeros([3, 1, 1, 10])
        feat_maps[1, :, :, :5] = 1.0
        feat_maps[2] = 1.0

        sparsity = population_sparsity(feat_maps)
        self.assertAlmostEqual(sparsity[0], 1.0)
        self.assertAlmostEqual(sparsity[1], 0.55, places = 1)
        self.assertAlmostEqual(sparsity[2], 0.0)
        

    def test_signal_power_AttributeError_no_repeat_col(self):
        df = pd.DataFrame(
            {
                'cell_1': np.arange(1000),
                'cell_2': np.arange(1000)
            }
        )
        
        with self.assertRaises(AttributeError):
            signal_power(df)

        
    def test_signal_power_KeyError_no_frame_col(self):
        df = pd.DataFrame(
            {
                'cell_1': np.arange(1000),
                'cell_2': np.arange(1000),
                'repeat': np.arange(1000)
            }
        )

        with self.assertRaises(KeyError):
            signal_power(df)        


if __name__ == '__main__':
    unittest.main(verbosity = 2)