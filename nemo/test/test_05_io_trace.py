''' 
Unit tests for the functions in data.io.trace.py.
'''


import csv
import os
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from nemo.data.io.trace import (
    compile_trial_avg_traces,
    load_trial_avg_trace_array
)


class TestIOTrace(unittest.TestCase):

    def test_compile_trial_avg_traces(self):
        n_frames = 5
        n_cells = 10
        write = np.random.randn(n_cells, n_frames) * 100

        with TemporaryDirectory() as tmp_dir:
            for i in range(n_cells):
                write_fpath = os.path.join(tmp_dir, 'cellID_0{}.txt'.format(i))

                with open(write_fpath, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['0{}.png'.format(j) for j in range(n_frames)])
                    writer.writerow(list(write[i]))

            traces, cell_ids = compile_trial_avg_traces(tmp_dir)
            self.assertAlmostEqual(np.sum(traces.to_numpy() - write), 0.0, places = 12)

    
    def test_load_trial_avg_trace_array(self):
        write = np.random.randn(10)

        with TemporaryDirectory() as tmp_dir:
            write_fpath = os.path.join(tmp_dir, 'cellID_0.txt')

            with open(write_fpath, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['0{}.png'.format(j) for j in range(10)])
                writer.writerow(list(write))
                
            for i in range(1, 10):
                read = load_trial_avg_trace_array(write_fpath, n_frames_in_time = i)
                self.assertAlmostEqual(np.sum(write[i-1:] - read), 0.0, places = 12)


    def test_load_trial_avg_trace_array_ValueError(self):
        write = np.random.randn(10)
        for i in range(-10, 1):
            with self.assertRaises(ValueError):
                load_trial_avg_trace_array(write, n_frames_in_time = i)



if __name__ == '__main__':
    unittest.main(verbosity = 2)