''' 
Unit tests for the classes in io.py
'''


import csv
import os
from tempfile import TemporaryDirectory
import unittest 

import cv2
import h5py 
import numpy as np

from nemo.data.io import (
    compile_trial_avg_traces,
    load_trial_avg_trace_array,
    read_frames,
    read_h5_as_array,
    save_vid_array_as_frames,
    write_csv
)


class TestIO(unittest.TestCase):

    def test_write_csv_list(self):
        with TemporaryDirectory() as tmp_dir:
            write = [list(range(5)) for _ in range(10)]
            write_fpath = os.path.join(tmp_dir, 'test_write_csv_list.txt')
            write_csv(write, write_fpath)

            with open(write_fpath, 'r') as f:
                reader = csv.reader(f)
                read = list(reader)
                
            compare = [
                all([w == int(r) for w, r in zip(w_list, r_list)]) 
                for w_list, r_list in zip(write, read)
            ]
            self.assertTrue(all(compare))


    def test_read_h5_as_array(self):
        write = np.random.randn(10000)
        
        with TemporaryDirectory() as tmp_dir:
            write_fpath = os.path.join(tmp_dir, 'test_read_h5_as_array.h5')
            
            with h5py.File(write_fpath, 'a') as h5file:
                h5file.create_dataset('test_array', data = write)

            read = read_h5_as_array(write_fpath)['test_array'][()]
            self.assertEqual(np.sum(write - read), 0.0)


    def test_save_vid_array_as_frames(self):
        write = np.random.uniform(0, 255, size = (10, 32, 64))
        write = np.uint8(write)

        with TemporaryDirectory() as tmp_dir:
            save_vid_array_as_frames([(write, tmp_dir)])
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


    def test_compile_trial_avg_traces(self):
        write = np.random.randn(100, 10)

        with TemporaryDirectory() as tmp_dir:
            for i in range(100):
                write_fpath = os.path.join(tmp_dir, 'cellID_{}.txt'.format(i))

                with open(write_fpath, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['0{}.png'.format(j) for j in range(10)])
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
            self.assertRaises(ValueError, load_trial_avg_trace_array, write, n_frames_in_time = i)

if __name__ == '__main__':
    unittest.main(verbosity = 2)