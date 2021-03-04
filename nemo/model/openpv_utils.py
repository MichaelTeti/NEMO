import numpy as np
from oct2py import octave


def get_pvp_weight_fpaths(ckpt_dir, fname_key = '*_W.pvp'):
    fpaths = glob(os.path.join(ckpt_dir, fname_key))
    fpaths.sort()

    return fpaths


def read_activity_file(model_activity_fpath, openpv_path = '/home/mteti/OpenPV/mlab/util'):
    '''
    Read in the <model_layer_name>_A.pvp file containing the activations / feature maps of each neuron.
    
    Args:
        model_activity_fpath (str): Path to the <model_layer_name>_A.pvp where the
            activations of each neuron are contained in.
        openpv_path (str): Path to the openpv matlab utility directory (*/OpenPV/mlab/util).
        
    Returns:
        acts (np.ndarray): A B x H x W x C array of feature maps. 
    '''
    
    # add OpenPV matlab utility dir to octave path
    octave.addpath(openpv_path)
    
    # Read in the activity file and get the number of batch samples
    # Should be a list of length batch size, where each item in the list is an array 
    # of shape (input_width / stride x) x (input height / stride y) x # Neurons
    # with the activations for that particular batch sample
    act_data = octave.readpvpfile(model_activity_fpath)
    n_batch = len(act_data)
    
    # concatenate across batch samples to produce a single array of shape 
    # batch size x (input width / stride x) x (input height / stride y) x # Neurons
    acts = np.concatenate([act_data[b]['values'][None, ...] for b in range(n_batch)], 0)
    
    # transpose the 2nd and third dimensions to make it B x H x W x # Neurons
    acts = acts.transpose([0, 2, 1, 3])
    
    return acts


def read_complex_cell_weight_files(fpaths, openpv_path = '/home/mteti/OpenPV/mlab/util'):
    octave.addpath(openpv_path)

    weights_agg = []
    for fpath in fpaths:
        # original shape: patch width x patch height x in channels x out channels
        # reshape to: out_channels x in channels x patch height x patch width
        weights = octave.readpvpfile(fpath)[0]['values'][0]
        weights = weights.transpose([3, 2, 1, 0])
        weights_agg.append(weights)

    return weights_agg