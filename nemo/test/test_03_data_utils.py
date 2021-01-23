''' 
Unit tests for the classes in io.py
'''


import csv
from multiprocessing import cpu_count
import os
from random import randint
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from nemo.data.utils import (
    add_string_to_fpaths,
    change_file_exts,
    get_fpaths_in_dir,
    multiproc 
)


class TestDataUtils(unittest.TestCase):

    def test_multiproc_ValueError1(self):

        def test_func(number1, number2):
            addition = number1 + number2

        n_cpus = cpu_count()
        with self.assertRaises(ValueError):
            multiproc(
                test_func,
                ['number1', 'number2'],
                n_workers = 10,
                number1 = list(range(100)),
                number2 = list(range(10))
            )     


    def test_multiproc_n_workers_geq_items(self):

        with TemporaryDirectory() as tmp_dir:
        
            def test_func(number1, number2, number3, number4, fname):
                with open(os.path.join(tmp_dir, fname), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow([number1, number1 + number2, number3, number4])

            list1 = list(range(10))
            list2 = list(range(-10, 0))
            list3 = [value1 + value2 for value1, value2 in zip(list1, list2)]
            fnames = ['0{}.txt'.format(i) for i in range(10)]

            multiproc(
                test_func,
                ['number1', 'number2', 'number3', 'fname'],
                n_workers = 20,
                number1 = list1,
                number2 = list2,
                number3 = list3,
                number4 = 1000,
                fname = fnames
            )

            read = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
            self.assertEqual(len(read), 10)
            
            for fpath in read:
                file_num = int(os.path.splitext(os.path.split(fpath)[1])[0])
                with open(fpath, 'r') as f:
                    data = list(csv.reader(f))[0]
                    data = [int(d) for d in data]
                    self.assertEqual(file_num, data[0])
                    self.assertEqual(data[1], data[2])
                    self.assertEqual(data[-1], 1000)


    def test_multiproc_n_workers_lt_items(self):

        with TemporaryDirectory() as tmp_dir:
        
            def test_func(list1, fnames):
                list1_len = len(list1)

                for val, fname in zip(list1, fnames):
                    with open(os.path.join(tmp_dir, fname), 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow([val, list1_len])

            list1 = list(range(10))
            fnames = ['0{}.txt'.format(i) for i in range(10)]

            multiproc(
                test_func,
                ['list1', 'fnames'],
                n_workers = 3,
                list1 = list1,
                fnames = fnames
            )

            read = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
            read.sort()
            self.assertEqual(len(read), 10)
            
            for fpath in read:
                file_num = int(os.path.splitext(os.path.split(fpath)[1])[0])
                with open(fpath, 'r') as f:
                    data = list(csv.reader(f))[0]
                    self.assertEqual(int(data[0]), file_num)


    def test_change_file_exts(self):
        fpaths = [
            os.path.join('{}'.format(i), '{}.test.jpg'.format(j)) 
            for i in range(10) 
            for j in range(10)
        ]
        fpaths_png = change_file_exts(fpaths)
        
        for fpath, fpath_png in zip(fpaths, fpaths_png):
            fpath, ext = os.path.splitext(fpath)
            fpath_png, ext_png = os.path.splitext(fpath_png)
            self.assertEqual(fpath, fpath_png)
            self.assertEqual(ext_png, '.png')


    def test_get_fpaths_in_dir(self):
        imgs = np.zeros([10 * 10 * 10, 32, 64])

        with TemporaryDirectory() as tmp_dir:
            fpaths = [
                os.path.join(tmp_dir, '{}'.format(i), '{}'.format(j), '{}.test.jpg'.format(k)) 
                for i in range(10) 
                for j in range(10)
                for k in range(10)
            ]

            for fpath, img in zip(fpaths, imgs):
                os.makedirs(os.path.split(fpath)[0], exist_ok = True)
                cv2.imwrite(fpath, img)

            fpaths_found = get_fpaths_in_dir(tmp_dir)
            
            for fpath_found in fpaths_found:
                self.assertTrue(fpath_found in fpaths)


    def test_get_fpaths_in_dir_key(self):
        with TemporaryDirectory() as tmp_dir:

            n_find = 0
            fpaths = []
            for i in range(10):
                for j in range(10):
                    keyword = 'find' if (i % 2 == 0 and j % 2 == 0) else 'miss'
                    n_find = n_find + 1 if keyword == 'find' else n_find
                    fpath = os.path.join(
                        tmp_dir,
                        '{}'.format(i),
                        '{}_{}'.format(j, keyword),
                        'test.png'
                    )
                    fpaths.append(fpath)
                    os.makedirs(os.path.split(fpath)[0], exist_ok = True)
                    cv2.imwrite(fpath, np.random.randn(32, 64))

            fpaths_found = get_fpaths_in_dir(tmp_dir, key = 'find')
            
            for fpath_found in fpaths_found:
                self.assertTrue(fpath_found in fpaths)


    def test_add_string_to_fpaths(self):
        fpaths = [
            os.path.join('{}'.format(i), '{}'.format(j), '{}.test.jpg'.format(k)) 
            for i in range(10) 
            for j in range(10)
            for k in range(10)
        ]
        fpaths_renamed = add_string_to_fpaths(fpaths, 'test_string')

        for fpath_renamed in fpaths_renamed:
            fpath_renamed_dir, fpath_renamed_fname = os.path.split(fpath_renamed)
            rename_undone = os.path.join(fpath_renamed_dir.split('test_string')[0], fpath_renamed_fname)
            self.assertTrue(rename_undone in fpaths)



if __name__ == '__main__':
    unittest.main(verbosity = 2)