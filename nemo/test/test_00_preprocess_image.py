''' 
Unit tests for the classes in data.preprocess.image.py
'''


import os
from random import randint
from tempfile import TemporaryDirectory
import unittest 

import cv2
import numpy as np

from nemo.data.preprocess.image import (
    center_crop,
    create_video_frame_sequences,
    max_min_scale,
    read_crop_write,
    read_downsample_write,
    read_pre_whiten_write,
    read_resize_write,
    read_smooth_write,
    read_whiten_write,
    resize_img,
    standardize_preds
)


class TestImagePreprocess(unittest.TestCase):

    def test_max_min_scale_range(self):
        '''
        Output vector values are in [0, 1] given random input.
        '''

        rand_vec = np.random.randn(10000) * 100
        rand_vec_scaled = max_min_scale(rand_vec)
        scaled_min = np.amin(rand_vec_scaled)
        scaled_max = np.amax(rand_vec_scaled)
        self.assertAlmostEqual(scaled_min, 0.0, places = 6)
        self.assertAlmostEqual(scaled_max, 1.0, places = 6)


    def test_max_min_scale_shape(self):
        '''
        Input and output shape are equal.
        '''

        rand_vec = np.random.randn(10000)
        rand_vec_scaled = max_min_scale(rand_vec)
        self.assertCountEqual(rand_vec_scaled.shape, rand_vec.shape)


    def test_create_video_frame_sequences_values(self):
        '''
        Output of tensor has correct placement of consecutive video frames.
        '''

        input_array = np.zeros([7, 32, 64])
        for i in range(7):
            input_array[i] = i
        
        output_array = create_video_frame_sequences(input_array, n_frames_in_time = 3)
        for i in range(5):
            self.assertEqual(np.sum(output_array[i, :, :, 0] - input_array[i]), 0)
            self.assertEqual(np.sum(output_array[i, :, :, 1] - input_array[i + 1]), 0)
            self.assertEqual(np.sum(output_array[i, :, :, 2] - input_array[i + 2]), 0)


    def test_create_video_frame_sequences_shape(self):
        '''
        Output shape is appropriate given input shape and n_frames_in_time.
        '''

        test_array = np.zeros([7, 32, 64])
        test_sequences = create_video_frame_sequences(test_array, n_frames_in_time = 3)
        self.assertCountEqual([5, 32, 64, 3], test_sequences.shape)


    def test_standardize_preds_range(self):
        '''
        Output has columns with mean close to 0 and std close to 1 given rand input.
        '''

        test_array = np.random.rand(1000, 32, 64) * 100
        test_array_standardized = standardize_preds(test_array)
        mean_diff = np.sum(np.mean(test_array_standardized, 0))
        std_diff = np.sum(np.std(test_array_standardized, 0) - np.ones([32, 64]))
        self.assertAlmostEqual(mean_diff, 0.0, places = 6)
        self.assertAlmostEqual(std_diff, 0.0, places = 6)


    def test_standardize_preds_shape(self):
        '''
        Input and output shapes are equal.
        '''

        test_array = np.random.randn(1000, 32, 64)
        test_array_standardized = standardize_preds(test_array)
        self.assertCountEqual(test_array_standardized.shape, test_array.shape)


    def test_resize_img_shape_same_aspect(self):
        '''
        Output shape equals desired output shape.
        '''

        img = np.uint8(np.random.uniform(0, 255, size = (64, 128)))
        img_resized = resize_img(img, 10, 20)
        self.assertCountEqual(img_resized.shape, [10, 20])


    def test_resize_img_shape_actual_aspect_gt_desired_aspect(self):
        '''
        Output shape equals desired output shape.
        '''

        img = np.uint8(np.random.uniform(0, 255, size = (64, 128)))
        img_resized = resize_img(img, 32, 45)
        self.assertCountEqual(img_resized.shape, [32, 45])


    def test_resize_img_shape_actual_aspect_lt_desired_aspect(self):
        '''
        Output shape equals desired output shape.
        '''

        img = np.uint8(np.random.uniform(0, 255, size = (64, 128)))
        img_resized = resize_img(img, 32, 80)
        self.assertCountEqual(img_resized.shape, [32, 80])


    def test_resize_img_values_resize_smaller(self):
        '''
        Output and input image have same shape.
        '''

        img = np.zeros([64, 128])
        img[27:37, 59:69] = 255
        img_resized = resize_img(img, 32, 64)
        img_unresized = resize_img(img_resized, 64, 128)
        diff = np.sum(img - img_unresized)
        self.assertEqual(diff, 0.0)


    def test_resize_img_values_resize_larger(self):
        '''
        Output and input image have same shape.
        '''

        img = np.zeros([64, 128])
        img[27:37, 59:69] = 255
        img_resized = resize_img(img, 128, 256)
        img_unresized = resize_img(img_resized, 64, 128)
        diff = np.sum(img - img_unresized)
        self.assertEqual(diff, 0.0)


    def test_read_resize_write_shape(self):
        '''
        Ouput shapes are equal to the desired shape.
        '''

        imgs = [np.uint8(np.random.uniform(0, 255, size = (64, 128))) for _ in range(10)]

        with TemporaryDirectory() as tmp_dir:
            fpaths = [os.path.join(tmp_dir, '0{}.png'.format(i)) for i in range(10)]
            write_fpaths = [os.path.splitext(fpath)[0] + '_write.png' for fpath in fpaths]

            for fpath, img in zip(fpaths, imgs):
                cv2.imwrite(fpath, img)

            for read_fpath, write_fpath in zip(fpaths, write_fpaths):
                read_resize_write(read_fpath, write_fpath, 32, 64, 0.0)

            for fpath in write_fpaths:
                resized_img = cv2.imread(fpath)
                self.assertCountEqual(resized_img.shape[:2], [32, 64])


    def test_read_resize_write_smaller_values(self):
        '''
        Check if the number different pixels between images is not more than were changed.
        '''

        imgs = np.zeros([10, 64, 128])
        perc_pixels_changed = (30 * 50) / imgs[0].size
        for i in range(10):
            start_r, end_r = i, i + 30
            start_c, end_c = i * 5, i * 5 + 50
            imgs[i, start_r:end_r, start_c:end_c] = 255

        with TemporaryDirectory() as tmp_dir:
            fpaths = [os.path.join(tmp_dir, '0{}.png'.format(i)) for i in range(10)]
            write_fpaths = [os.path.splitext(fpath)[0] + '_write.png' for fpath in fpaths]

            for fpath, img in zip(fpaths, imgs):
                cv2.imwrite(fpath, img)

            for read_fpath, write_fpath in zip(fpaths, write_fpaths):
                read_resize_write(read_fpath, write_fpath, 32, 64, 0.0)

            for read_fpath, write_fpath in zip(fpaths, write_fpaths):
                original_img = cv2.cvtColor(cv2.imread(read_fpath), cv2.COLOR_BGR2GRAY)
                resized_img = cv2.cvtColor(cv2.imread(write_fpath), cv2.COLOR_BGR2GRAY)
                unresized_img = cv2.resize(resized_img, (128, 64))
                diff = original_img - unresized_img
                perc_pixels_diff = np.sum(diff != 0) / diff.size
                self.assertTrue(perc_pixels_diff < perc_pixels_changed)      


    def test_read_resize_write_larger_values(self):
        '''
        Check if the number different pixels between images is not more than were changed.
        '''

        imgs = np.zeros([10, 32, 64])
        perc_pixels_changed = (10 * 20) / imgs[0].size
        for i in range(10):
            start_r, end_r = i, i + 10
            start_c, end_c = i * 3, i * 3 + 20
            imgs[i, start_r:end_r, start_c:end_c] = 255

        with TemporaryDirectory() as tmp_dir:
            fpaths = [os.path.join(tmp_dir, '0{}.png'.format(i)) for i in range(10)]
            write_fpaths = [os.path.splitext(fpath)[0] + '_write.png' for fpath in fpaths]

            for fpath, img in zip(fpaths, imgs):
                cv2.imwrite(fpath, img)

            for read_fpath, write_fpath in zip(fpaths, write_fpaths):
                read_resize_write(read_fpath, write_fpath, 64, 128, 0.0)

            for read_fpath, write_fpath in zip(fpaths, write_fpaths):
                original_img = cv2.cvtColor(cv2.imread(read_fpath), cv2.COLOR_BGR2GRAY)
                resized_img = cv2.cvtColor(cv2.imread(write_fpath), cv2.COLOR_BGR2GRAY)
                unresized_img = cv2.resize(resized_img, (64, 32))
                diff = original_img - unresized_img
                perc_pixels_diff = np.sum(diff != 0) / diff.size
                self.assertTrue(perc_pixels_diff < perc_pixels_changed) 


    def test_read_resize_write_n_files(self):
        '''
        Number of files in directory equals number of files resized and written.
        '''

        imgs = [np.uint8(np.random.uniform(0, 255, size = (64, 128))) for _ in range(10)]

        with TemporaryDirectory() as tmp_dir:
            fpaths = [os.path.join(tmp_dir, '0{}.png'.format(i)) for i in range(10)]
            write_fpaths = [os.path.splitext(fpath)[0] + '_write.png' for fpath in fpaths]

            for fpath, img in zip(fpaths, imgs):
                cv2.imwrite(fpath, img)

            for read_fpath, write_fpath in zip(fpaths, write_fpaths):
                read_resize_write(read_fpath, write_fpath, 32, 64, 0.0)
            
            for fpath in write_fpaths:
                self.assertTrue(os.path.split(fpath)[1] in os.listdir(tmp_dir))


    def test_read_resize_write_ValueError(self):
        '''
        Negative aspect_ratio_tol raises ValueError.
        '''

        imgs = [np.uint8(np.random.uniform(0, 255, size = (64, 128))) for _ in range(10)]

        with TemporaryDirectory() as tmp_dir:
            fpaths = [os.path.join(tmp_dir, '0{}.png'.format(i)) for i in range(10)]
            write_fpaths = [os.path.splitext(fpath)[0] + '_write.png' for fpath in fpaths]

            for fpath, img in zip(fpaths, imgs):
                cv2.imwrite(fpath, img)

            with self.assertRaises(ValueError):
                read_resize_write(fpaths[0], write_fpaths[0], 32, 64, -10.0)


    def test_read_resize_write_over_tol(self):
        '''
        Files not written if desired and actual aspect ratios over aspect_ratio_tol. 
        '''

        img = np.uint8(np.random.uniform(0, 255, size = (64, 32)))

        with TemporaryDirectory() as tmp_dir:
            read_fpath = os.path.join(tmp_dir, 'dataset' 'test_img.png')
            write_fpath = os.path.join(tmp_dir, 'dataset_resize', 'test_img_resize.png')
            cv2.imwrite(read_fpath, img)

            read_resize_write(read_fpath, write_fpath, 32, 64, aspect_ratio_tol = 1.0)
            self.assertTrue('dataset_resize' not in os.listdir(tmp_dir))


    def test_read_crop_write_shape(self):
        '''
        Cropped images have appropriate shapes.
        '''

        imgs = [np.uint8(np.random.uniform(0, 255, size = (64, 128))) for _ in range(10)]

        with TemporaryDirectory() as tmp_dir:
            fpaths = [os.path.join(tmp_dir, '0{}.png'.format(i)) for i in range(10)]
            write_fpaths = [os.path.splitext(fpath)[0] + '_write.png' for fpath in fpaths]

            for fpath, img in zip(fpaths, imgs):
                cv2.imwrite(fpath, img)

            for read_fpath, write_fpath in zip(fpaths, write_fpaths):
                read_crop_write(read_fpath, write_fpath, 32, 64)

            for fpath in write_fpaths:
                resized_img = cv2.imread(fpath)
                self.assertCountEqual(resized_img.shape[:2], [32, 64])


    def test_read_crop_write_n_files(self):
        '''
        Number of files in directory equals number of files resized and written.
        '''

        imgs = [np.uint8(np.random.uniform(0, 255, size = (64, 128))) for _ in range(10)]

        with TemporaryDirectory() as tmp_dir:
            fpaths = [os.path.join(tmp_dir, '0{}.png'.format(i)) for i in range(10)]
            write_fpaths = [os.path.splitext(fpath)[0] + '_write.png' for fpath in fpaths]

            for fpath, img in zip(fpaths, imgs):
                cv2.imwrite(fpath, img)

            for read_fpath, write_fpath in zip(fpaths, write_fpaths):
                read_crop_write(read_fpath, write_fpath, 32, 64)
            
            for fpath in write_fpaths:
                self.assertTrue(os.path.split(fpath)[1] in os.listdir(tmp_dir))


    def test_center_crop_values(self):
        '''
        Cropped image has appropriate values relative to input image.
        '''

        img = np.uint8(np.random.uniform(0, 255, size = (64, 128)))
        img_cropped = center_crop(img, 62, 126)
        diff = np.sum(img[1:-1, 1:-1] - img_cropped)
        self.assertEqual(diff, 0.0)


    def test_center_crop_shape_even2odd(self):
        '''
        Cropped image has appropriate shape from even to odd dims.
        '''

        img = np.uint8(np.random.uniform(0, 255, size = (64, 128)))
        img_cropped = center_crop(img, 35, 79)
        self.assertCountEqual(img_cropped.shape, [35, 79])


    def test_center_crop_shape_odd2even(self):
        '''
        Cropped image has appropriate shape from odd to even dims.
        '''

        img = np.uint8(np.random.uniform(0, 255, size = (65, 129)))
        img_cropped = center_crop(img, 32, 64)
        self.assertCountEqual(img_cropped.shape, [32, 64])


    def test_read_downsample_write_shape(self):
        '''
        Downsampled image has appropriate shape given downsample factor.
        '''

        img = np.uint8(np.random.uniform(0, 255, size = (64, 128)))

        with TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, 'test_img.png')
            write_fpath = os.path.join(tmp_dir, 'test_img_write.png')
            cv2.imwrite(fpath, img)
            read_downsample_write(fpath, write_fpath, 2, 2)
            img_downsampled = cv2.imread(write_fpath)
            self.assertCountEqual(img_downsampled.shape[:2], [32, 64])


    def test_read_downsample_write_values(self):
        '''
        Downsampled image has appropriate values corresponding to input image.
        '''

        img = np.uint8(np.random.uniform(0, 255, size = (64, 128)))

        with TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, 'test_img.png')
            write_fpath = os.path.join(tmp_dir, 'test_img_write.png')
            cv2.imwrite(fpath, img)
            read_downsample_write(fpath, write_fpath, 2, 2)
            img_downsampled = cv2.imread(write_fpath)
            img_downsampled = cv2.cvtColor(img_downsampled, cv2.COLOR_BGR2GRAY)
            diff = np.sum(img[::2, ::2] - img_downsampled)
            self.assertEqual(diff, 0.0)

    
    def test_read_downsample_write_n_files(self):
        '''
        Number of files in directory equals number of files resized and written.
        '''

        imgs = [np.uint8(np.random.uniform(0, 255, size = (64, 128))) for _ in range(10)]

        with TemporaryDirectory() as tmp_dir:
            fpaths = [os.path.join(tmp_dir, '0{}.png'.format(i)) for i in range(10)]
            write_fpaths = [os.path.splitext(fpath)[0] + '_write.png' for fpath in fpaths]

            for fpath, img in zip(fpaths, imgs):
                cv2.imwrite(fpath, img)

            for read_fpath, write_fpath in zip(fpaths, write_fpaths):
                read_downsample_write(read_fpath, write_fpath, 2, 2)
            
            for fpath in write_fpaths:
                self.assertTrue(os.path.split(fpath)[1] in os.listdir(tmp_dir))


    def test_read_smooth_write_shape(self):
        '''
        Smoothed images have same shape as original images.
        '''

        img = np.uint8(np.random.uniform(0, 255, size = (64, 128)))

        with TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, 'test_img.png')
            write_fpath = os.path.join(tmp_dir, 'test_img_write.png')
            cv2.imwrite(fpath, img)
            read_smooth_write(fpath, write_fpath)
            img_smoothed = cv2.imread(write_fpath)
            img_smoothed = cv2.cvtColor(img_smoothed, cv2.COLOR_BGR2GRAY)
            self.assertCountEqual(img_smoothed.shape[:2], [64, 128])


    def test_read_smooth_write_n_files(self):
        '''
        Number of files read equals number of files resized and written.
        '''

        imgs = [np.uint8(np.random.uniform(0, 255, size = (64, 128))) for _ in range(10)]

        with TemporaryDirectory() as tmp_dir:
            fpaths = [os.path.join(tmp_dir, '0{}.png'.format(i)) for i in range(10)]
            write_fpaths = [os.path.splitext(fpath)[0] + '_write.png' for fpath in fpaths]

            for fpath, img in zip(fpaths, imgs):
                cv2.imwrite(fpath, img)

            for read_fpath, write_fpath in zip(fpaths, write_fpaths):
                read_smooth_write(read_fpath, write_fpath)
            
            for fpath in write_fpaths:
                self.assertTrue(os.path.split(fpath)[1] in os.listdir(tmp_dir))


    def test_read_pre_whiten_write_shape(self):
        '''
        Pre-whitened images have same shape as input images.
        '''

        img = np.uint8(np.random.uniform(0, 255, size = (64, 128)))

        with TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, 'test_img.png')
            write_fpath = os.path.join(tmp_dir, 'test_img_write.png')
            cv2.imwrite(fpath, img)
            read_pre_whiten_write(fpath, write_fpath)
            img_pre_whitened = cv2.imread(write_fpath)
            img_pre_whitened = cv2.cvtColor(img_pre_whitened, cv2.COLOR_BGR2GRAY)
            self.assertCountEqual(img_pre_whitened.shape[:2], [64, 128])


    def test_read_pre_whiten_write_n_files(self):
        '''
        Number of files read equals number of files resized and written.
        '''

        imgs = [np.uint8(np.random.uniform(0, 255, size = (64, 128))) for _ in range(10)]

        with TemporaryDirectory() as tmp_dir:
            fpaths = [os.path.join(tmp_dir, '0{}.png'.format(i)) for i in range(10)]
            write_fpaths = [os.path.splitext(fpath)[0] + '_write.png' for fpath in fpaths]

            for fpath, img in zip(fpaths, imgs):
                cv2.imwrite(fpath, img)

            for read_fpath, write_fpath in zip(fpaths, write_fpaths):
                read_pre_whiten_write(read_fpath, write_fpath)
            
            for fpath in write_fpaths:
                self.assertTrue(os.path.split(fpath)[1] in os.listdir(tmp_dir))


    def test_read_whiten_write_shape(self):
        '''
        Whitened images have same shape as original images.
        '''

        imgs = [np.uint8(np.random.uniform(0, 255, size = (32, 64))) for _ in range(10)]

        with TemporaryDirectory() as tmp_dir:
            fpaths = [os.path.join(tmp_dir, '0{}.png'.format(i)) for i in range(10)]
            write_fpaths = [os.path.join(tmp_dir, '0{}_write.png'.format(i)) for i in range(10)]

            for fpath, img in zip(fpaths, imgs):
                cv2.imwrite(fpath, img)

            read_whiten_write([fpaths], [write_fpaths])
            
            for fpath in write_fpaths:
                whitened = cv2.imread(fpath)
                whitened = cv2.cvtColor(whitened, cv2.COLOR_BGR2GRAY)
                self.assertCountEqual(whitened.shape[:2], [32, 64])


    def test_read_whiten_write_n_files(self):
        '''
        Number of files read equals number of files resized and written.
        '''

        imgs = [np.uint8(np.random.uniform(0, 255, size = (32, 64))) for _ in range(10)]

        with TemporaryDirectory() as tmp_dir:
            fpaths = [os.path.join(tmp_dir, '0{}.png'.format(i)) for i in range(10)]
            write_fpaths = [os.path.join(tmp_dir, '0{}_write.png'.format(i)) for i in range(10)]

            for fpath, img in zip(fpaths, imgs):
                cv2.imwrite(fpath, img)

            read_whiten_write([fpaths], [write_fpaths])
            
            for fpath in write_fpaths:
                fname = os.path.split(fpath)[1]
                self.assertTrue(fname in os.listdir(tmp_dir))



if __name__ == '__main__':
    unittest.main(verbosity = 2)