import numpy as np


def lifetime_sparsity(feat_maps, eps = 1e-12):
    '''
    Computes lifetime sparsity for each neuron as in 
    https://www.biorxiv.org/content/biorxiv/suppl/2018/06/29/359513.DC1/359513-1.pdf.

    Args:
        feat_maps (np.ndarray): The feature maps of shape B x H x W x # Neurons. 
            If H and W are equal to 1, then the lifetime sparsity is computed over
            the batch. Otherwise, it is computed over the batch and spatial dims.

    Returns lifetime_sparsity (np.ndarray): A vector of shape # Neurons with the 
        lifetime sparsity of each neuron. 
    '''

    n_neurons = feat_maps.shape[-1]
    feat_maps = feat_maps.reshape([-1, n_neurons])
    B = feat_maps.shape[0]
    
    sum_square = np.sum(feat_maps, 0) ** 2
    square_sum = np.sum(feat_maps ** 2, 0)

    return (1 - (1 / B) * (sum_square + eps) / (square_sum + eps)) / (1 - 1 / B)


def mean_activations(acts):
    '''
    Read in the activations of each neuron, average them over space and batch,
    sort them based on the average.
    
    Args:
        acts (np.ndarray): Array of feature maps of shape B x H x W x # Neurons.
        
    Returns: 
        sorted_acts (np.ndarray): A 4D of shape Batch x Input H x Input W x # Neurons,
            that is sorted descending at the last axis by each neuron's mean activation.
        sorted_mean_acts(np.ndarray): A 1D array of size # Neurons containing each 
            neuron's mean activation and sorted descending by this value.
        sorted_inds (list): A list of length n_neurons corresponding to sorted_mean_acts and 
            consisting of the original index of the neuron with that sorted activation.
    '''

    # get the mean activation for each neuron across batch and spatial dimensions
    # and sort them descending based on this mean and keep the neuron index too
    mean_acts = np.mean(acts, (0, 1, 2))
    se_acts = np.std(acts, (0, 1, 2)) / np.sqrt(acts[..., 0].size)
    inds = list(range(mean_acts.size))
    sorted_inds = [ind for _,ind in sorted(zip(mean_acts, inds), reverse = True)]
    sorted_mean_acts = mean_acts[sorted_inds]
    sorted_se_acts = se_acts[sorted_inds]
    
    return sorted_mean_acts, sorted_se_acts, sorted_inds


def population_sparsity(feat_maps, eps = 1e-12):
    '''
    Computes population sparsity for each stimulus in a batch as in 
    https://www.biorxiv.org/content/biorxiv/suppl/2018/06/29/359513.DC1/359513-1.pdf.

    Args:
        feat_maps (np.ndarray): The feature maps of shape B x H x W x # Neurons. 
            If H and W are equal to 1, then the population sparsity is computed over
            the neurons. Otherwise, it is computed over the neurons and spatial dims.

    Returns lifetime_sparsity (np.ndarray): A vector of shape B with the 
        population sparsity of each batch sample.
    '''

    B = feat_maps.shape[0]
    feat_maps = feat_maps.reshape([B, -1])
    n_neurons = feat_maps.shape[1]

    sum_square = np.sum(feat_maps, 1) ** 2 
    square_sum = np.sum(feat_maps ** 2, 1)

    return (1 - (1 / n_neurons) * (sum_square + eps) / (square_sum + eps)) / (1 - 1 / n_neurons)


def signal_power(df):
    ''' 
    Compute signal power of a neuron's responses over trials as in 
    https://www.frontiersin.org/articles/10.3389/fncom.2016.00010/full

    Args:
        df (pd.DataFrame): DataFrame with a 'repeat' and 'frame' column and columns for 
            each cell's responses.

    Returns:
        power (pd.Series): Series of length len(df.columns) - 2 with each cell's signal 
            power.
    '''

    try:
        N = df.repeat.max() + 1
        var_sum = df.drop(columns = 'repeat').groupby('frame').sum().var(axis = 0)
        sum_var = df.drop(columns = 'frame').groupby('repeat').var().sum(axis = 0)
    except (AttributeError, KeyError):
        print("df should have a column called 'repeat' and one called 'frame'.")
        raise 
    
    return (var_sum - sum_var) / (N * (N - 1))


def cc_norm(true, pred, SP):
    ''' 
    Compute CC_norm between true and predicted responses as in 
    https://www.frontiersin.org/articles/10.3389/fncom.2016.00010/full

    Args:
        true (np.ndarray): A vector of true trial-averaged responses for a single neuron.
        pred (np.ndarray): A vector of predicted responses corresponding to true.
        SP (float): Signal power for this neuron.

    Returns:
        cc_norm (float): Correlation of pred with true given relative to amount of 
            explainable variance.
    '''

    cov = np.cov(true, pred)
    cc_norm = cov[0, 1] / np.sqrt(cov[1, 1] * SP)
    
    return cc_norm


def cc_max(true, SP):
    ''' 
    Computes CC_max of measured responses as in 
    https://www.frontiersin.org/articles/10.3389/fncom.2016.00010/full

    Args:
        true (pd.DataFrame): A DataFrame with N columns where each column
            is a trial averaged response vector for a single neuron.
        SP (pd.Series): Signal power for each neuron in true. 

    Returns:
        cc_max (pd.Series): the maximum correlation coefficient between the recorded firing 
            rate and the best prediction that a perfect model could theoretically achieve 
            for each neuron.
    '''

    return np.sqrt(SP / true.var(axis = 0))