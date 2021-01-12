import cv2
import numpy as np


def resize_and_keep_aspect(img, desired_height, desired_width):
    '''
    Resizes an image to a new size while keeping the original aspect ratio by
    cropping the larger side.

    Args:
        img (np.ndarray): Color or grayscale image to resize.
        desired_height (int): Height to resize img to.
        desired_width (int): Width to resize image to.

    Returns:
        img_resized (np.ndarray): The resized image with the same aspect ratio as img.
    '''

    assert desired_height > 0
    assert desired_width > 0

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

    assert crop_height > 0
    assert crop_width > 0

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

    assert len(imgs.shape) == 2

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


def create_temporal_design_mat(vid_frame_array, n_frames_in_time = 9):
    '''
    Creates a design mat composed of flattened video frame sequences. 

    Args:
        vid_frame_array (np.ndarray): A N x H x W array of the video frames in order.
        n_frames_in_time (int): The number of frames to look at at a time.

    Returns:
        design_mat (np.ndarray): A design matrix of flattened video frame sequences. 
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
