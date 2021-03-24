'''
Unit tests for the classes in model.utils.py
'''


from argparse import Namespace
import os
from tempfile import TemporaryDirectory
import unittest 

import numpy as np
import torch

from nemo.model.utils import (
    cv_splitter_video,
    save_args,
    shuffle_design_mat,
    to_tensor
)


class TestModelUtils(unittest.TestCase):

    def test_cv_splitter_video_ValueError(self):
        for i in range(-10, 1):
            self.assertRaises(ValueError, cv_splitter_video, 100, i)

    
    def test_cv_splitter_video_n_folds(self):
        for n_folds in range(1, 11):
            cv_inds = cv_splitter_video(100, n_folds = n_folds)
            self.assertEqual(len(cv_inds), n_folds)

    
    def test_cv_splitter_video_n_samples_per_fold(self):
        for n_folds in range(1, 11):
            cv_inds = cv_splitter_video(100, n_folds = n_folds)
            n_samples_fold = [t_inds.size + v_inds.size == 100 for t_inds, v_inds in cv_inds]
            self.assertTrue(all(n_samples_fold))


    def test_cv_splitter_video_n_unique_samples_per_fold(self):
        for n_folds in range(1, 11):
            cv_inds = cv_splitter_video(100, n_folds = n_folds)
            n_unique_fold = [np.unique(np.concatenate([t_inds, v_inds])).size == 100 for t_inds, v_inds in cv_inds]
            self.assertTrue(all(n_unique_fold))


    def test_shuffle_design_mat_shape(self):
        preds = np.random.randn(100, 64)
        responses = np.random.randn(100)
        shuffled_preds, shuffled_responses = shuffle_design_mat(preds, responses)
        self.assertCountEqual(responses.shape, shuffled_responses.shape)
        self.assertCountEqual(preds.shape, shuffled_preds.shape)


    def test_shuffle_design_mat_values(self):
        preds = np.random.rand(100, 64)
        responses = np.random.rand(100)
        shuffled_preds, shuffled_responses = shuffle_design_mat(preds, responses)
        pred_diff = np.sum(preds - shuffled_preds)
        response_diff = np.sum(responses - shuffled_responses)
        self.assertNotEqual(pred_diff, 0.0)
        self.assertNotEqual(response_diff, 0.0)


    def test_save_args(self):
        args = {
                'arg1': 3.0,
                'arg2': 'test',
                'arg3': ['value1', 'value2', 'value3'],
                'arg4': False
        }
        args = Namespace(**args)

        with TemporaryDirectory() as tmp_dir:
            save_args(args, tmp_dir)
            self.assertTrue('args.txt' in os.listdir(tmp_dir))


    @unittest.skipIf(not torch.cuda.is_available(), 'skipping because cuda not available')
    def test_to_tensor_device(self):
        data = np.arange(10)
        data_tensor = to_tensor(data, dev = 0)
        self.assertEqual(data_tensor.device.index, 0)
        self.assertCountEqual(data_tensor.cpu().numpy(), data)


    def test_to_tensor_no_device(self):
        data = np.arange(10)
        data_tensor = to_tensor(data)
        self.assertEqual(data_tensor.device.type, 'cpu')
        self.assertCountEqual(data_tensor.numpy(), data)


if __name__ == '__main__':
    unittest.main(verbosity = 2)