import csv
from multiprocessing import Process, cpu_count
import os

import h5py
import numpy as np


def multiproc(func, iterator_keys, n_workers = 4, return_list = False, **kwargs):
    '''
    A general function to use multiprocessing to perform other functions that do not return anything.

    Args:
        func (function): A previously defined function.
        iterator_keys (str): A list of keys in kwargs whose values are the lists to divide up
            among n_workers in separate calls to func.
        n_workers (int): The number of processes to use. Be careful with this.
        return_list (bool): If True and a worker has a list of length 1, then it will keep the item
            in the list. Otherwise, it will take the item outside of the list.
        kwargs: keyword arguments to func. The items specified by iterator_keys should be lists.

    Returns:
        None
    '''

    # check if all lists to be divided up are same length
    n_inputs = len(kwargs[iterator_keys[0]])
    if not all([len(kwargs[key]) == n_inputs for key in iterator_keys]):
        raise ValueError('lists corresponding to iterator_keys should all have same length.')

    # find out how many inputs to each worker
    if n_inputs < n_workers: n_workers = n_inputs
    inputs_per_worker = int(np.ceil(n_inputs / n_workers))

    # loop over inputs and divide up between the workers for each process
    procs = []
    for worker_num, input_num in enumerate(range(0, n_inputs, inputs_per_worker)):
        kwarg_inputs = kwargs.copy()
        start_ind = input_num
        end_ind = input_num + inputs_per_worker

        for key in iterator_keys:
            kwarg_inputs[key] = kwarg_inputs[key][start_ind:end_ind]

            if len(kwarg_inputs[key]) == 1 and not return_list:
                kwarg_inputs[key] = kwarg_inputs[key][0]

        process = Process(target = func, kwargs = kwarg_inputs)
        procs.append(process)
        process.start()

    for proc in procs:
        process.join()


def change_file_exts(fpaths, desired_ext = '.png'):
    '''
    Changes the file extension for every fpath in a list of fpaths.

    Args:
        fpaths (list): List of fpaths or fnames to modify.
        desired_ext (str): Desired extension of the files. Should begin with a period,
            but one will be added if not there.

    Returns:
        fpaths_new (list): List of fpaths corresponding to fpaths input with desired_ext
            the extension of the files.
    '''
        
    if desired_ext[0] != '.': 
        desired_ext = '.' + desired_ext

    fpaths_new = [os.path.splitext(fpath)[0] + desired_ext for fpath in fpaths]
    
    return fpaths_new


def get_fpaths_in_dir(dir, key = None):
    '''
    Walk down a directory structure and return a list of all fpaths.

    Args:
        dir (str): The directory to start at and traverse down.
        key (str): An optional key. If given, will only return fpaths if key is
                   either in one of the parent dirs or the fname itself.

    Returns:
        fpaths (list): List of fpaths.
    '''

    fpaths = []
    for root, _, files in os.walk(dir):
        files.sort()

        for file in files:
            fpath = os.path.join(root, file)
            if key and key not in fpath: continue
            fpaths.append(fpath)

    return fpaths


def add_string_to_fpaths(fpaths, string):
    '''
    Adds a string to the last dir given a file's fpath.

    Args:
        fpaths (list): A list of paths to files.
        string (str): A string to add to the last dir in fpaths.
            e.g. add_string_to_fpaths(['home/user/files/file.txt'], '_shortened')
            would return ['home/user/files_shortened/file.txt']

    Returns:
        fpaths_modified (list): fpaths modified by the string
    '''
    
    if type(fpaths) not in [list, tuple]:
        raise TypeError('fpaths must be a list or tuple.')

    fpaths = [os.path.join(os.path.split(fpath)[0] + str(string), os.path.split(fpath)[1]) for fpath in fpaths]

    return fpaths


def get_img_frame_names(n_frames):
    '''
    Generate numbers up to n_frames with leading zeros (useful for saving video array as images).

    Args:
        n_frames (int): The number of frames in the array (i.e. number to count up to).

    Returns:
        frame_numbers (list of strings)
    '''

    if n_frames <= 0:
        raise ValueError('n_frames should be > 0.')

    n_decimals = len(str(n_frames))
    
    return ['0' * (n_decimals - len(str(i))) + str(i) for i in range(n_frames)]


def download_experiment_data(ids, boc):
    '''
    Download AllenSDK experiment container files.

    Args:
        ids (list): experiment ids to download data.
        boc (BrainObservatoryCache object)
        
    Returns:
        None
    '''

    for id in ids:
        boc.get_ophys_experiment_data(id)


def write_csv(items, fpath, mode = 'w'):
    '''
    Write a list to a .csv file.

    Args:
        items (list): A list of values (e.g. strings, floats, ints, etc.) or lists.
        fpath (str): Desired fpath for the .csv file.
        mode (str): Write mode. See https://docs.python.org/3.6/library/functions.html#open
              for details.

    Returns:
        None
    '''

    with open(fpath, mode) as f:
        writer = csv.writer(f, delimiter = ',')
        for item in items:
            if type(item) != list: item = [item]
            writer.writerow(item)


def read_h5_as_array(fpath):
    '''
    Read a .h5 file and return the items as a dictionary of np.ndarrays.

    Args:
        fpath (str): The path to the .h5 file.

    Returns:
        data (dict): A dictionary of keys and their corresponding arrays.
    '''

    with h5py.File(fpath, 'r+') as h5file:
        data = {}
        for key in list(h5file.keys()):
            data[key] = h5file[key][()]

    return data


def read_csv(fpath, remove_header = False, remove_footer = False, mode = 'r',):
    '''
    Read a .csv file and return the items as a list.

    Args:
        fpath (str): The path to the .csv file.
        remove_header (bool): True to return lines 1-n.
        remove_footer (bool): True to return lines 0-(n-1).
        mode (str): Mode to open the file in.

    Returns:
        A list of items (e.g. strs, floats, ints, lists, etc.)
    '''

    with open(fpath, mode) as f:
        reader = csv.reader(f, delimiter = ',')
        data = []

        try:
            for row_num, row in enumerate(reader[:-1] if remove_footer else reader):
                if row_num == 0 and remove_header: continue
                data.append(row[0] if len(row) == 1 else row)
        except:
            pass

    return data


def monitor_coord_to_image_ind(x_cm, y_cm, monitor_w_cm = 51.91, monitor_h_cm = 32.44):
    '''
    Takes in the monitor coordinates in centimeters returned by 
    dataset.get_pupil_location(as_spherical = False),
    where (0, 0) is the center of the monitor, and turns those into proportions 
    where (0, 0) is the top left corner of the monitor and (1, 1) is the bottom
    right corner.
    
    Args:
        x_cm (np.ndarray, float): Horizontal location on monitor in cm. 
        y_cm (np.ndarray, float): Vertical location on monitor in cm.
        monitor_w_cm (float): Width of monitor in cm.
        monitor_h_cm (float): Height of monitor in cm.

    Returns:
        x_img: Horizontal location on monitor as proportion of monitor width.
            0 is the left of the monitor, and 1 is the right.
        y_img: Vertical location on monitor as proportion of monitor height.
            0 is the top of the monitor and 1 is the bottom.
    '''
    
    if type(x_cm) not in [np.ndarray, float, int]:
        raise TypeError
    if type(y_cm) not in [np.ndarray, float, int]:
        raise TypeError

    y_cm *= -1 
    x_img = np.round(((monitor_w_cm / 2) + x_cm) / monitor_w_cm, 5)
    y_img = np.round(((monitor_h_cm / 2) + y_cm) / monitor_h_cm, 5)
    
    if type(x_img) == np.ndarray:
        if np.amin(x_img) < 0 or np.amax(x_img) >= 1:
            raise ValueError
    else:
        if x_img < 0 or x_img >= 1:
            raise ValueError
        
    if type(y_img) == np.ndarray:
        if np.amin(y_img) < 0 or np.amax(y_img) >= 1:
            raise ValueError
    else:
        if y_img < 0 or y_img >= 1:
            raise ValueError
    
    return x_img, y_img