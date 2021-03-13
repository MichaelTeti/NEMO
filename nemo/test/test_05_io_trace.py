''' 
Unit tests for the functions in data.io.trace.py.
'''

import os
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from nemo.data.io.trace import (
    read_neural_data_file
)


class TestIOTrace(unittest.TestCase):

    def test_read_neural_data_file_df_values_stim_None(self):
        with TemporaryDirectory() as tmp_dir:
            df = pd.DataFrame(
                {
                    'stimulus': ['stim1', 'stim2', 'stim3'],
                    'frame': list(range(3))
                }
            )
            fpath = os.path.join(tmp_dir, '000.txt')
            df.to_csv(fpath, index = False)
            df_read, _ = read_neural_data_file(fpath)

            self.assertTrue(df.equals(df_read))


    def test_read_neural_data_file_cell_id(self):
        with TemporaryDirectory() as tmp_dir:
            df = pd.DataFrame(
                {
                    'stimulus': ['stim1', 'stim2', 'stim3'],
                    'frame': list(range(3))
                }
            )
            fpath = os.path.join(tmp_dir, '000.txt')
            df.to_csv(fpath, index = False)
            _, cell_id = read_neural_data_file(fpath)
            
            self.assertEqual(cell_id, '000')


    def test_read_neural_data_file_df_values_stim_specified_and_present(self):
        with TemporaryDirectory() as tmp_dir:
            df = pd.DataFrame(
                {
                    'stimulus': ['stim1', 'stim2', 'stim3'],
                    'frame': list(range(3))
                }
            )
            fpath = os.path.join(tmp_dir, '000.txt')
            df.to_csv(fpath, index = False)
            df_read, _ = read_neural_data_file(fpath, stimuli = ['stim1'])
            
            self.assertTrue(df_read.equals(df[df.stimulus == 'stim1']))

        
    def test_read_neural_data_file_df_values_stim_specified_and_not_present(self):
        with TemporaryDirectory() as tmp_dir:
            df = pd.DataFrame(
                {
                    'stimulus': ['stim1', 'stim2', 'stim3'],
                    'frame': list(range(3))
                }
            )
            fpath = os.path.join(tmp_dir, '000.txt')
            df.to_csv(fpath, index = False)
            df_read, _ = read_neural_data_file(fpath, stimuli = ['stim10'])
            
            self.assertTrue(df_read.equals(pd.DataFrame()))


if __name__ == '__main__':
    unittest.main(verbosity = 2)