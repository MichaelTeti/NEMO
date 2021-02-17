Here, we use LCA to learn a dictionary of features with convolutional sparse coding. 

# Training 
## Creating Input Filepath Lists
First, it is necessary to create .txt files with the file paths of the video frames the LCA model will be 
trained on in order. To do this, we use the 
[make_image_fpath_list.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/lca_scripts/make_image_fpath_list.py) 
script. First, move to the [lca_scripts](https://github.com/MichaelTeti/NEMO/tree/main/scripts/lca_scripts) directory (```cd ../../scripts/lca_scripts/```), then use the following command:
```
python3 make_image_fpath_list.py \
    ../../data/ILSVRC2015/Data/VID/train/ \
    ../../experiments/lca_dictionary_learning_shared/ \
    --key _resized \
    --n_frames_in_time 9
```
The first two arguments are required. The first one is the directory to start at when recursing down through the file
structure looking for images, and the second one is the directory where the .txt files containing the file paths 
will be saved. The third argument is optional and indicates the filter used when determining which sub-directories 
located in ```../../data/ILSVRC2015/Data/VID/train/``` and containing video frames should be included in the .txt files. 
The last argument determines the number of consecutive frames that will be used as a single input, which also determines 
how many .txt files are written out. It is possible to just write all the frames in order on a single .txt file, 
and read in 9 consecutive frames in at a time from it, but there may be some drastic changes from one video to the next within a single input in that case. Here, we make sure that every single input contains only video frames from one video.   
  
## Running the Model
The model parameters are described in the [learn_imagenet_dict.lua](https://github.com/MichaelTeti/NEMO/blob/main/experiments/lca_dictionary_learning_shared/learn_imagenet_dict.lua) script. The model is convolutional, as indicated by the line ```local sharedWeights = true;```, with a stride of 1 (```strideX``` and ```strideY``` are set to 1). The features are 17 x 17, as determined by ```patchSizeY``` and ```patchSizeX```), and the images are of shape 32 x 64 as described by ```inputHeight``` and ```inputWidth```, respectively. The batch size, indicated by ```nbatch```, is set to 256, and the line ```temporalPatchSize = 9``` means that each of the 256 mini-batch samples are composed of 9 consecutive video frames. We use a soft threshold with lambda equal to 0.1 (which is increased during training). The [train_lca.sh](https://github.com/MichaelTeti/NEMO/blob/main/experiments/lca_dictionary_learning_shared/train_lca.sh) script is used to run the model in an HPC setting. 

# Analysis
## Running the Analysis Script
The [analysis.py](https://github.com/MichaelTeti/NEMO/blob/main/experiments/lca_dictionary_learning_shared/analysis.py) script is used to analyze the model. Specifically, the script is used to visualize:
  * The model's features
  * The reconstruction error over training
  * The sparsity over training 
  * The reconstructions and inputs

The arguments to this script are as follows:

```
python3 analysis.py --help
usage: analysis.py [-h] [--openpv_path OPENPV_PATH] [--no_features]
                   [--no_recons] [--no_probes] [--no_activity]
                   [--weight_fpath_key WEIGHT_FPATH_KEY]
                   [--rec_layer_key REC_LAYER_KEY]
                   [--input_layer_key INPUT_LAYER_KEY] --activity_fpath
                   ACTIVITY_FPATH --sparse_activity_fpath
                   SPARSE_ACTIVITY_FPATH --probe_dir PROBE_DIR
                   [--l2_probe_key L2_PROBE_KEY]
                   [--firm_thresh_probe_key FIRM_THRESH_PROBE_KEY]
                   [--energy_probe_key ENERGY_PROBE_KEY]
                   [--adaptive_ts_probe_key ADAPTIVE_TS_PROBE_KEY]
                   [--display_period_length DISPLAY_PERIOD_LENGTH]
                   [--n_display_periods N_DISPLAY_PERIODS]
                   [--plot_individual_probes]
                   ckpt_dir save_dir

positional arguments:
  ckpt_dir              Path to the OpenPV checkpoint.
  save_dir              Where to save these analyses and write out plots.

optional arguments:
  -h, --help            show this help message and exit
  --openpv_path OPENPV_PATH
                        Path to *OpenPV/mlab/util
  --no_features         If specified, will not plot the features.
  --no_recons           If specified, will not plot the reconstructions and
                        inputs.
  --no_probes           If specified, will not plot the probes.
  --no_activity         If specified, will not plot mean activations, mean
                        sparsity, or mean activity.

feature visualization:
  Arguments for the view_complex_cell_strfs function.

  --weight_fpath_key WEIGHT_FPATH_KEY
                        A key to help find the desired _W.pvp files in the
                        ckpt directory, since there may be multiple in the
                        same one for some models.

reconstruction visualization:
  The arguments for the view_reconstructions function.

  --rec_layer_key REC_LAYER_KEY
                        A key to help find the .pvp files written out by the
                        reconstruction layer in the checkpoint directory.
  --input_layer_key INPUT_LAYER_KEY
                        A key to help find the .pvp files written out by the
                        input layer in the checkpoint directory.

mean activations:
  The arguments to plot the mean activations and mean sparstiy.

  --activity_fpath ACTIVITY_FPATH
                        The path to the <model_layer_name>_A.pvp file.

number active:
  The arguments for the plot_num_neurons_active function.

  --sparse_activity_fpath SPARSE_ACTIVITY_FPATH
                        Path to the <model_layer_name>.pvp file.

probes:
  The arguments for the plot_probes.py script.

  --probe_dir PROBE_DIR
                        The directory where the probes are written out.
  --l2_probe_key L2_PROBE_KEY
                        A key to help filter out the L2 Probe files in
                        probe_dir (e.g. "*L2Probe*").
  --firm_thresh_probe_key FIRM_THRESH_PROBE_KEY
                        A key to help filter out the Firm Thresh files in
                        probe_dir.
  --energy_probe_key ENERGY_PROBE_KEY
                        A key to help filter out the Energy files in the
                        probe_dir.
  --adaptive_ts_probe_key ADAPTIVE_TS_PROBE_KEY
                        A key to help filter out the adaptive timescale probes
                        in probe_dir.
  --display_period_length DISPLAY_PERIOD_LENGTH
                        The length of the display period.
  --n_display_periods N_DISPLAY_PERIODS
                        How many display periods to show starting from the end
                        and going backward.
  --plot_individual_probes
                        If specified, will plot each probe .txt file
                        individually as well as moving average.
```  

## Outputs of the Analysis Script
Below are some examples of the things the analysis script writes out.

### Model Features
The analysis script writes a .gif of the model's features in ```save_dir```. For example, the features of the model trained here can be viewed below.   
![features.gif](https://github.com/MichaelTeti/NEMO/blob/main/experiments/lca_dictionary_learning_shared/figures/features.gif)  

### Inputs and Reconstructions
We can also visualize the reconstruction performance of the model by viewing the inputs and reconstructions. The top row is the input. The second row is the reconstruction, and the third row is the difference between the two.   
![inputs_and_recons.gif](https://github.com/MichaelTeti/NEMO/blob/main/experiments/lca_dictionary_learning_shared/figures/inputs_and_recons.gif)   

### Model Activity
Here, we can visualize the mean activations across spatial and batch dimensions for each neuron in the model. They are sorted from highest to lowest.   
![mean_acts.png](https://github.com/MichaelTeti/NEMO/blob/main/experiments/lca_dictionary_learning_shared/figures/mean_activations.png)   
   
### Objective Function
The plot below shows the objective function during one of the runs.   
![objective.png](https://github.com/MichaelTeti/NEMO/blob/main/experiments/lca_dictionary_learning_shared/figures/minimum_energy.png)   
   
We can also observe just the L2 reconstruction error over ther run. 
![reconstruction.png](https://github.com/MichaelTeti/NEMO/blob/main/experiments/lca_dictionary_learning_shared/figures/minimum_l2.png)   
   
And finally the L1 sparsity penalty term over the run.   
![sparsity.png](https://github.com/MichaelTeti/NEMO/blob/main/experiments/lca_dictionary_learning_shared/figures/minimum_firm_thresh.png)
