''' 
Unit tests for the classes in data.preprocess.trace.py
'''

import unittest

import numpy as np

from nemo.data.preprocess.trace import normalize_traces


class TestTracePreprocess(unittest.TestCase):

    def test_normalize_traces_range(self):
        '''
        Output of normalize_traces has mean 0 and max 1.
        '''

        test_traces = np.random.rand(10000) * 100
        test_traces_norm = normalize_traces(test_traces)
        norm_mean = np.mean(test_traces_norm)
        norm_max = np.amax(np.absolute(test_traces_norm))
        self.assertAlmostEqual(norm_mean, 0.0, places = 6)
        self.assertAlmostEqual(norm_max, 1.0, places = 6)


    def test_normalize_traces_shape(self):
        '''
        Output and input shape of normalize_traces match. 
        '''

        test_traces = np.random.randn(10000)
        test_traces_norm = normalize_traces(test_traces)
        self.assertCountEqual(test_traces_norm.shape, test_traces.shape)


if __name__ == '__main__':
    unittest.main(verbosity = 2)