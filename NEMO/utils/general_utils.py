from multiprocessing import Process, cpu_count
import os

import csv
import h5py
import numpy as np
import pandas as pd


def multiproc(func, iterator_key, n_workers = 4, **kwargs):
    '''
    A general function to use multiprocessing to perform other functions that do not return anything.
    Args:
        func (function): A previously defined function.
        iterator_key (str): A key in kwargs whose value is the list to divide up
            among n_workers in separate calls to func.
        n_workers (int): The number of processes to use.
        kwargs: keyword arguments to func.
    Returns:
        None
    '''

    assert(n_workers <= cpu_count()), \
        'n_workers ({}) should be <= the value returned with cpu_count ({})'.format(n_workers, cpu_count())

    # find out how many inputs to each worker
    procs = []
    inputs = kwargs[iterator_key]
    n_inputs = len(inputs)
    if n_inputs < n_workers: n_workers = n_inputs
    inputs_per_worker = int(np.ceil(n_inputs / n_workers))

    # loop over inputs and divide up between the workers for each process
    for worker_num, input_num in enumerate(range(0, n_inputs, inputs_per_worker)):
        func_inputs = inputs.copy()
        kwarg_inputs = kwargs.copy()
        start_ind = input_num
        end_ind = input_num + inputs_per_worker
        func_inputs = func_inputs[start_ind:end_ind]
        kwarg_inputs[iterator_key] = func_inputs
        process = Process(target = func, kwargs = kwarg_inputs)
        procs.append(process)
        process.start()

    for proc in procs:
        process.join()


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


def read_csv(fpath, remove_header = False, mode = 'r', remove_footer = False):
    '''
    Read a .csv file and return the items as a list.
    Args:
        fpath (str): The path to the .csv file.
        remove_header (bool): True to return lines 2-n.
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


def str2float_list(input_list, start_ind = 0):
    '''
    Takes a list of lists composed of numeric strings and turns the strings into floats.
    Args:
        input_list (list): List of lists to operate on.
        start_ind (int): Index of each sub-list to start turning into floats. Default is 0.
    Returns:
        float_list (list): List corresponding to input_list with float values instead of str.
    '''

    if type(start_ind) not in [int, np.uint8, np.int16, np.int32, np.int64]:
        raise TypeError('start_ind should be of type int in function str2float_list, \
            but is of type {}.'.format(type(start_ind)))

    float_list = [[float(val) if val_num >= start_ind else val for val_num, val in enumerate(sublist)] for sublist in input_list]
    return float_list


def write_h5(fpath, keys, mats, mode = 'a'):
    '''
    Write numpy arrays to a .h5 file.
    Args:
        fpath (str): The desired path to the file.
        keys (str, tuple of strs, or list of strs): The desired key names for each item in mats.
        mats (np.ndarray or list of np.ndarrays): The array(s) to save under key(s) in the .h5 file.
        mode (str): Write mode. Refer to http://docs.h5py.org/en/stable/high/file.html for more details.
    Returns:
        None
    '''
    f = h5py.File(fpath, mode)

    if type(keys) in [list, tuple]:
        if type(mats) == np.ndarray:
            raise TypeError('Multiple keys given as type {}, but mats is type np.ndarray.'.format(type(keys)))
        if len(keys) != len(mats):
            raise IndexError('len(keys) ({}) does not match len(mats) ({})'.format(len(keys), len(mats)))
        for key, mat in zip(keys, mats):
            f.create_dataset(key, data = mat)

    elif type(keys) == str:
        if type(data) != np.ndarray:
            raise TypeError('one key given as type str. Expected a single np.ndarray for mats, but got type {}.'.format(type(mats)))
        f.create_dataset(keys, data = mats)

    f.close()


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
    if type(desired_ext) != str:
        raise TypeError('Arg desired_ext should be of type string, but is {}.'.format(type(desired_ext)))
    if desired_ext[0] != '.': desired_ext = '.' + desired_ext

    fpaths_new = [os.path.splitext(fpath)[0] + desired_ext for fpath in fpaths]
    return fpaths_new



def read_h5_as_array(fpath):
    '''
    Read a .h5 file and return the items as a dictionary of np.ndarrays.
    Args:
        fpath (str): The path to the .h5 file.
    Returns:
        data (dict): A dictionary of keys and their corresponding arrays.
    '''
    f = h5py.File(fpath, 'r+')
    data = {}
    for key in list(f.keys()):
        data[key] = np.array(f[key])

    f.close()
    return data


def find_common_vals_in_lists(*lists):
    if len(lists) <= 1:
        raise ValueError('find_common_vals_in_lists needs at least 2 lists.')
    if not all([type(l) == list for l in lists]):
        raise TypeError('Inputs to find_common_vals_in_lists must all be lists.')

    for i, l in enumerate(lists):
        if i == 0:
            common = set(l)
            continue

        common = common.intersection(set(l))

    return list(common)



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

    if key and type(key) != str:
        raise TypeError('key argument should be of type str, but is of type {}.'.format(type(key)))

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
        fpaths (str): A list of paths to files.
        string (str): A string to add to the last dir in fpaths.
            e.g. add_string_to_fpaths(['home/user/files/file.txt'], '_shortened')
            would return ['home/user/files_shortened/file.txt']
    Returns:
        fpaths_modified (list): fpaths modified by the string
    '''
    if type(string) != str:
        raise TypeError('Argument string should be of type str, but is of type {}.'.format(type(string)))
    if type(fpaths) != list:
        raise TypeError('Argument fpaths should be of type list, but is of type {}.'.format(type(fpaths)))

    fpaths = [os.path.join(os.path.split(fpath)[0] + string, os.path.split(fpath)[1]) for fpath in fpaths]

    return fpaths


def get_intersection_col_vals(datasets, col_name):
    col_vals = [dataset[col_name] for dataset in datasets]
    for dset_num in range(1, len(col_vals)):
        if dset_num == 1:
            common_col_vals = pd.merge(col_vals[0], col_vals[1], how = 'inner')
        else:
            common_col_vals = pd.merge(common_col_vals, col_vals[dset_num], how = 'inner')

    return common_col_vals[col_name]
