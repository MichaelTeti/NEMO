import numpy as np


def normalize_traces(traces):
    '''
    Zero-mean and scale traces.
    
    Args:
        traces (np.ndarray): The array of unscaled fluorescence trace values.
        
    Returns:
        traces_scaled (np.ndarray): The array of fluorescence traces with range [-1, 1].
    '''
    
    traces /= np.amax(np.absolute(traces), 0)
    
    return traces