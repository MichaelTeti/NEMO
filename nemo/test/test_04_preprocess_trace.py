''' 
Unit tests for the classes in data.preprocess.trace.py
'''

import unittest

import numpy as np

from nemo.data.preprocess.trace import normalize_traces


class TestTracePreprocess(unittest.TestCase):

    def test_normalize_traces_range_single_cell(self):
        '''
        Output of normalize_traces has absolute max 1 for a single cell.
        '''

        test_traces = np.random.rand(10000) * 100
        test_traces_norm = normalize_traces(test_traces)
        norm_max = np.amax(np.absolute(test_traces_norm))
        self.assertAlmostEqual(norm_max, 1.0, places = 6)


    def test_normalize_traces_shape_single_cell(self):
        '''
        Output and input shape of normalize_traces match for a single cell. 
        '''

        test_traces = np.random.randn(10000)
        test_traces_norm = normalize_traces(test_traces)
        self.assertCountEqual(test_traces_norm.shape, test_traces.shape)


    def test_normalize_traces_shape_multi_cell(self):
        '''
        Output and input shape of normalize_traces match for multiple cells.
        '''

        test_traces = np.random.randn(10000, 10)
        test_traces_norm = normalize_traces(test_traces)
        self.assertCountEqual(test_traces_norm.shape, test_traces.shape)


    def test_normalize_traces_range_single_cell(self):
        '''
        Output of normalize_traces has absolute max 1 for multiple cells.
        '''

        test_traces = np.random.rand(10000, 10) * 100
        test_traces_norm = normalize_traces(test_traces)
        norm_max = np.amax(np.absolute(test_traces_norm), 0)
        
        for i in range(10):
            self.assertAlmostEqual(norm_max[i], 1.0, places = 12)



if __name__ == '__main__':
    unittest.main(verbosity = 2)