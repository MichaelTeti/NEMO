''' 
Unit tests for the classes in data.preprocess.trace.py
'''

import unittest

import numpy as np
import pandas as pd

from nemo.data.preprocess.trace import (
    aggregate_cell_data,
    compute_trial_avgs,
    normalize_traces
)


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


    def test_compute_trial_avgs_AttributeError(self):
        df = pd.DataFrame({'stimulus': [0, 1], 'frame': [1, 2]})
        
        with self.assertRaises(AttributeError):
            compute_trial_avgs(df)


    def test_compute_trial_avgs_KeyError(self):
        df = pd.DataFrame({'dff': [0, 1]})
        
        with self.assertRaises(KeyError):
            compute_trial_avgs(df) 


    def test_compute_trial_avgs_shape(self):
        df = pd.DataFrame(
            {
                'dff': [1, 2, 3, 4],
                'frame': [0, 1, 0, 1],
                'stimulus': ['stim'] * 4
            }
        )
        df_avg = compute_trial_avgs(df)
        self.assertCountEqual(df_avg.shape, [2, 3])


    def test_compute_trial_avgs_values(self):
        df = pd.DataFrame(
            {
                'dff': [1, 2, 3, 4],
                'frame': [0, 1, 0, 1],
                'stimulus': ['stim'] * 4
            }
        )
        df_avg = compute_trial_avgs(df)
        
        self.assertCountEqual(df_avg.stimulus, ['stim'] * 2)
        self.assertCountEqual(df_avg.frame, [0, 1])
        self.assertCountEqual(df_avg.dff, [2, 3])


    def test_aggregate_cell_data_col_names_cell_ids(self):
        dfs = [
            pd.DataFrame(
                {
                    'a': [0, 1, 2],
                    'b': [1, 2, 3]
                }
            ),
            pd.DataFrame(
                {
                    'a': [0, 1, 2],
                    'b': [2, 3, 4]
                }
            )
        ]
        dfs_merged = aggregate_cell_data(
            dfs, 
            cell_ids = ['001', '000'], 
            keep_cols = ['b']
        )
        self.assertCountEqual(dfs_merged.columns, ['a', '001_b', '000_b'])


    def test_aggregate_cell_data_col_names_no_cell_ids(self):
        dfs = [
            pd.DataFrame(
                {
                    'a': [0, 1, 2],
                    'b': [1, 2, 3]
                }
            ),
            pd.DataFrame(
                {
                    'a': [0, 1, 2],
                    'b': [2, 3, 4]
                }
            )
        ]
        dfs_merged = aggregate_cell_data(dfs, keep_cols = ['b'])
        self.assertCountEqual(dfs_merged.columns, ['a', '0_b', '1_b'])


    def test_aggregate_cell_data_col_names_no_keep_cols(self):
        dfs = [
            pd.DataFrame(
                {
                    'a': [0, 1, 2],
                    'b': [1, 2, 3]
                }
            ),
            pd.DataFrame(
                {
                    'a': [0, 1, 2],
                    'b': [2, 3, 4]
                }
            )
        ]
        dfs_merged = aggregate_cell_data(dfs)
        self.assertCountEqual(dfs_merged.columns, ['a', 'b'])


    def test_aggregate_cell_data_df_shape_keep_cols(self):
        dfs = [
            pd.DataFrame(
                {
                    'a': [0, 1, 2],
                    'b': [1, 2, 3]
                }
            ),
            pd.DataFrame(
                {
                    'a': [0, 1, 2],
                    'b': [2, 3, 4]
                }
            )
        ]
        dfs_merged = aggregate_cell_data(dfs, keep_cols = ['b'])
        self.assertCountEqual(dfs_merged.shape, [3, 3])


    def test_aggregate_cell_data_df_shape_no_keep_cols(self):
        dfs = [
            pd.DataFrame(
                {
                    'a': [0, 1, 2],
                    'b': [1, 2, 3]
                }
            ),
            pd.DataFrame(
                {
                    'a': [0, 1, 2],
                    'b': [2, 3, 4]
                }
            )
        ]
        dfs_merged = aggregate_cell_data(dfs)
        self.assertCountEqual(dfs_merged.shape, [6, 2])


if __name__ == '__main__':
    unittest.main(verbosity = 2)