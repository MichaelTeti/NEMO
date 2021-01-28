''' 
Unit tests for the functions in data.io.image.py
'''


import os
from tempfile import TemporaryDirectory
import unittest 

import cv2
import numpy as np

from nemo.data.io.image import (
    read_frames,
    save_vid_array_as_frames,
)


class TestIOImage(unittest.TestCase):

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



if __name__ == '__main__':
    unittest.main(verbosity = 2)