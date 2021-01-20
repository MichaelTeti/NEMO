''' 
Unit tests for the classes in io.py
'''


from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from nemo.data.utils import (
    multiproc 
)


class TestDataUtils(unittest.TestCase):

    def test_multiproc(self):
        imgs = [np.random.randn(32, 64) for _ in range(10)]
        fpaths = ['0{}.png'.format(i) for i in range(10)]

        def save_img(fpath_and_img):
            cv2.imwrite(fpath_and_img[0], fpath_and_img[0])

        with TemporaryDirectory() as tmp_dir:
            



if __name__ == '__main__':
    unittest.main(verbosity = 2)