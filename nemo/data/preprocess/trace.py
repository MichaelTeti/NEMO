import numpy as np


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