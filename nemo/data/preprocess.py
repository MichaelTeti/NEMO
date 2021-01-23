import os

import cv2
import numpy as np


def resize_img(img, desired_height, desired_width):
    '''
    Resizes an image to a new size without warping by cropping the smaller dim.

    Args:
        img (np.ndarray): Color or grayscale image to resize.
        desired_height (int): Height to resize img to.
        desired_width (int): Width to resize image to.

    Returns:
        img_resized (np.ndarray): The resized image with the same aspect ratio as img.
    '''

    if desired_height <= 0 or desired_width <= 0:
        raise ValueError('desired_height and desired_width must be non-zero and positive.')

    h, w = img.shape[:2]
    desired_aspect = desired_width / desired_height
    actual_aspect = w / h

    if desired_aspect > actual_aspect:
        crop_height = h - int(w * desired_height / desired_width)
        if crop_height == 1:
            img = img[:-1]
        elif crop_height % 2 != 0:
            img = img[crop_height // 2:h - (crop_height // 2 + 1)]
        else:
            img = img[crop_height // 2: h - crop_height // 2]
    else:
        crop_width = w - int(h * desired_width / desired_height)
        if crop_width == 1:
            img = img[:, :-1]
        elif crop_width % 2 != 0:
            img = img[:, crop_width // 2:w - (crop_width // 2 + 1)]
        else:
            img = img[:, crop_width // 2:w - crop_width // 2]

    img_resized = cv2.resize(img, (desired_width, desired_height))

    return img_resized


def center_crop(img, crop_height, crop_width):
    '''
    Takes a center crop from an image given crop height and width.

    Args:
        img (np.ndarray): Image to crop with either 1, 3, or 4 channels.
        crop_height (int): Height of the center crop.
        crop_width (int): Width of the center crop.
        
    Returns:
        img_cropped (np.ndarray): The cropped image.
    '''

    if crop_height <= 0 or crop_width <= 0:
        raise ValueError('crop_height and crop_width must be non-zero and positive')

    crop_y = img.shape[0] // 2 - crop_height // 2
    crop_x = img.shape[1] // 2 - crop_width // 2
    img_cropped = img[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

    return img_cropped


def spatial_whiten(imgs, return_zca_mat = False, full_matrix = False):
    '''
    Function to ZCA whiten images.

    Args:
        imgs (np.ndarray): Matrix of samples x pixel values to whiten.
        return_zca_mat (bool): True to return the transformation matrix only, False
            to return the whitened images only (default).
        full_matrix (bool): Whether to use all of the SVD components (refer to
            https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
            for help).

    Returns:
        ZCAMatrix (np.ndarray) or whitened image matrix (np.ndarray) depending on return_zca_mat.
    '''

    if len(imgs.shape) != 2:
        raise ValueError('imgs must be a 2D array.')

    imgs = imgs - np.mean(imgs, 0)
    U,S,V = np.linalg.svd(np.matmul(imgs, imgs.transpose()), full_matrices = full_matrix)
    epsilon = 1e-6
    ZCAMatrix = np.matmul(U, np.matmul(np.diag(1.0 / np.sqrt(S + epsilon)), U.transpose()))

    if return_zca_mat:
        return ZCAMatrix
    else:
        return np.matmul(ZCAMatrix, imgs)


def max_min_scale(array, eps = 1e-8):
    '''
    Linearly scale the values in an array to the range [0, 1].

    Args:
        array (np.ndarray): The n-dimensional array to scale.
        eps (float): A small value added to the denominator to avoid division by 0.

    Returns:
        np.ndarray with max value of 1.0 and min value of 0.0.
    '''

    return (array - np.amin(array)) / (np.amax(array) - np.amin(array) + eps)


def make_lgn_freq_filter(x_dim, y_dim, f_0 = 300):
    '''
    Frequency spectrum filter used in
    https://www.frontiersin.org/articles/10.3389/fncir.2019.00013/full

    Args:
        x_dim (int): Width of the images.
        y_dim (int): Height of the images.
        f_0 (int): Cutoff for cycles/s. Should be about 0.4 * min(x_dim, y_dim)
            for biologically-realistic acuity.

    Returns:
        A filter for the fft transformed image.
    '''

    y_vals = np.arange(-y_dim // 2, y_dim // 2)
    x_vals = np.arange(-x_dim // 2, x_dim // 2)
    freqs_x, freqs_y = np.meshgrid(x_vals, y_vals)
    freq_grid = np.sqrt(freqs_x ** 2 + freqs_y ** 2)
    freq_filter = freq_grid * np.exp(-(freq_grid / f_0) ** 4)

    return freq_filter


def create_video_frame_sequences(vid_frame_array, n_frames_in_time = 9):
    '''
    Takes in an array of consecutive video frames and concatenates consecutive frames 
        to make samples of subsequences. 

    Args:
        vid_frame_array (np.ndarray): A N x H x W array of the video frames in order.
        n_frames_in_time (int): The number of frames to have in a single sample subsequence.

    Returns:
        design_mat (np.ndarray): An (N-n_frames_in_time+1) x H x W x n_frames_in_time array 
            of flattened video frame sequences. 
    '''
    
    n_frames, h, w = vid_frame_array.shape
    cat_list = [vid_frame_array[i:n_frames - (n_frames_in_time - i - 1)][..., None] for i in range(n_frames_in_time)] 
    design_mat = np.concatenate(cat_list, axis = 3)
    
    return design_mat


def normalize_traces(traces):
    '''
    Zero-mean and scale traces.
    
    Args:
        traces (np.ndarray): The array of unscaled fluorescence trace values.
        
    Returns:
        traces_scaled (np.ndarray): The array of fluorescence traces with zero mean and range [-1, 1].
    '''
    
    traces -= np.mean(traces)
    traces /= np.amax(np.absolute(traces))
    
    return traces


def standardize_preds(design_mat, mean_vec = None, std_vec = None, eps = 1e-12):
    '''
    Standardizes each predictor variable in the design matrix with the statistics of that column. 
    
    Args:
        design_mat (np.ndarray): An N-dimensional array where the first dimension indexes
            the data samples. 
        mean_vec (np.ndarray): An (N-1)-dimensional array with the mean of each dimension
            taken along the first dimension of design_mat. 
        std_vec (np.ndarray): An (N-1)-dimensional array with the std of each dimension
            taken along the first dimension of design_mat.
        eps (float): A scalar added to the elements in std_vec to avoid (unlikely)
            division by zero. 
            
    Returns:
        design_mat (np.ndarray): The design_mat with standardized predictors. 
    '''
    
    # if mean_vec and/or std_vec not given, calculate from the data
    if not mean_vec:
        mean_vec = np.mean(design_mat, 0)
    if not std_vec:
        std_vec = np.std(design_mat, 0)
        
    design_mat = (design_mat - mean_vec) / (std_vec + eps)
    
    return design_mat


def read_resize_write(read_fpaths, write_fpaths, desired_height, desired_width, aspect_ratio_tol = 0.26):
    '''
    Read in images based on fpaths, resize, and save in a new fpath.
    
    Args:
        read_fpaths (list): List of the fpaths to read the pre-resized images from.
        write_fpaths (list): List of the fpaths to write the post-resized images to.
        desired_height (int): Height to resize each image.
        desired_width (int): Width to resize each image.
        aspect_ratio_tol (float): Discard images if absolute value between
            original aspect ratio and resized aspect ratio >= aspect_ratio_tol
            to help avoid cropping more than a desired amount.
            
    Returns:
        None
    '''

    if aspect_ratio_tol < 0:
        raise ValueError('aspect_ratio_tol should be >= 0.')
    
    for fpath, new_fpath in zip(read_fpaths, write_fpaths):
        # read in the image
        img = cv2.imread(fpath)
        
        # calculate aspect ratio
        original_aspect = img.shape[1] / img.shape[0]
        
        # if aspect is not within aspect_ratio_tol of desired aspect, remove new dir then continue
        desired_aspect = desired_width / desired_height
        
        if abs(desired_aspect - original_aspect) > aspect_ratio_tol:
            if os.path.isdir(os.path.split(new_fpath)[0]): os.rmdir(os.path.split(new_fpath)[0])
            continue

        # resize the images
        img = resize_img(img, desired_height, desired_width)
        
        # save the resized image
        cv2.imwrite(new_fpath, img)


def read_crop_write(read_fpaths, write_fpaths, crop_height, crop_width):
    '''
    Read in images, crop them, and resave them.
    
    Args:
        read_fpaths (list): List of the fpaths to read the pre-cropped images from.
        write_fpaths (list): List of the fpaths to write the post-cropped images to.
        crop_height (int): Height of the cropped image.
        crop_width (int): Width of the cropped image.
        
    Returns:
        None
    '''

    for fpath, save_fpath in zip(read_fpaths, write_fpaths):
        # read in the image
        img = cv2.imread(fpath)
        
        # check if the image is smaller than the specified crop dims
        if img.shape[0] <= crop_height or img.shape[1] <= crop_width:
            if os.path.isdir(os.path.split(save_fpath)[0]): os.rmdir(os.path.split(save_fpath)[0])
            continue

        # crop it and save
        img = center_crop(img, crop_height, crop_width)
        cv2.imwrite(save_fpath, img)


def read_downsample_write(read_fpaths, write_fpaths, downsample_h = 2, downsample_w = 2):
    '''
    Reads in images from fpaths, subsamples, and resaves them.
    
    Args:
        read_fpaths (list): List of the fpaths to read the pre-downsampled images from.
        write_fpaths (list): List of the fpaths to write the post-downsampled images to.
        downsample_h (int): The factor to downsample the image height by.
        downsample_w (int): The factor to downsample the image width by.
        
    Returns:
        None
    '''

    for fpath, save_fpath in zip(read_fpaths, write_fpaths):
        # read in the video frames and downsample
        img = cv2.imread(fpath)[::downsample_h, ::downsample_w]
        
        # save the downsampled frame
        cv2.imwrite(save_fpath, img)


def read_smooth_write(read_fpaths, write_fpaths, neighborhood = 9, sigma_color = 75, sigma_space = 75):
    '''
    Read in images based on fpaths, smooth, and save in a new fpath.

    Args:
        read_fpaths (list): List of the fpaths to read the pre-smoothed images from.
        write_fpaths (list): List of the fpaths to write the post-smoothed images to.
        neighborhood (int): Diameter of the pixel neighborhood.
        sigma_color (float): Larger values mean larger differences in colors can be mixed together.
        sigma_space (float): Larger values mean larger differences in space can be mixed together.

    Returns:
        None
    '''

    for fpath, new_fpath in zip(read_fpaths, write_fpaths):
        # read in the image
        img = cv2.imread(fpath)

        # smooth the image
        img = cv2.bilateralFilter(
            img,
            d = neighborhood,
            sigmaColor = sigma_color,
            sigmaSpace = sigma_space
        )

        # save the resized image
        cv2.imwrite(new_fpath, img)


def read_pre_whiten_write(read_fpaths, write_fpaths, f_0 = None):
    '''
    Read in images based on fpaths, whiten, and save in a new fpath.
    
    Args:
        read_fpaths (list): List of the fpaths to read the non pre-whitened images from.
        write_fpaths (list): List of the fpaths to write the pre-whitened images to.
        f_0 (int): Cycles/s desired.
        
    Returns:
        None
    '''

    for fpath_num, (fpath, new_fpath) in enumerate(zip(read_fpaths, write_fpaths)):
        # read in the image and make grayscale
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[0], img.shape[1]

        # make the filter
        if fpath_num == 0:
            if not f_0: f_0 = np.ceil(min(h, w) * 0.4)
            ffilter = make_lgn_freq_filter(w, h, f_0 = f_0)

        # fft transform on image, filter, and go back to image
        img_fft = np.fft.fft2(img)
        img_fft *= ffilter
        img_rec = np.absolute(np.fft.ifft2(img_fft))

        # scale image to [0, 255] and write image
        img_scaled = max_min_scale(img_rec) * 255
        cv2.imwrite(new_fpath, img_scaled)


def read_whiten_write(read_fpaths, write_fpaths, full_svd = False, scale_method = 'video'):
    '''
    Read in images based on fpaths, resize, and save in a new fpath.
    
    Args:
        read_fpaths (list): List of the fpaths to read the non whitened images from.
        write_fpaths (list): List of the fpaths to write the whitened images to.
        full_svd (bool): If True, use all SVD components.
        scale_method (str): Whether to scale each frame from max/min of the video or only that frame.

    Returns:
        None
    '''

    for read_dir, save_dir in zip(read_fpaths, write_fpaths):
        for fpath_num, fpath in enumerate(read_dir):
            # read in the image and make grayscale
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

            # make a matrix to store the images if first image
            if fpath_num == 0:
                h, w = img.shape
                img_mat = np.zeros([len(read_dir), h * w])

            # add the image to the mat
            img_mat[fpath_num] = img.flatten()

        # do whitening
        img_mat_whitened = spatial_whiten(img_mat, full_matrix = full_svd)

        # scale frames based on max/min from entire video
        if scale_method == 'video':
            img_mat_whitened = max_min_scale(img_mat_whitened) * 255

        # save images
        for save_fpath_num, (new_fpath, flattened_img) in enumerate(zip(save_dir, img_mat_whitened)):
            # scale frame based on max/min from that frame if specified
            if scale_method == 'frame':
                flattened_img = max_min_scale(flattened_img) * 255

            img_rec = flattened_img.reshape([h, w])
            if save_fpath_num == 0: os.makedirs(os.path.split(new_fpath)[0], exist_ok = True)
            cv2.imwrite(new_fpath, img_rec)