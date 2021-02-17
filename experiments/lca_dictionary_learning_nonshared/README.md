Here, we use LCA to learn a dictionary of simple cell features. 

# Training 
## Creating Input Filepath Lists
First, it is necessary to create .txt files with the file paths of the video frames the LCA model will be 
trained on in order. To do this, we use the 
[make_image_fpath_list.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/lca_scripts/make_image_fpath_list.py) 
script. First, move to the [lca_scripts](https://github.com/MichaelTeti/NEMO/tree/main/scripts/lca_scripts) directory (```cd ../../scripts/lca_scripts/```), then use the following command (from the scripts/lca_scripts dir):
```
python make_image_fpath_list.py \
    ../../data/ILSVRC2015/Data/VID/train/ \
    ../../experiments/lca_dictionary_learning_nonshared/ \
    --key _resized \
    --n_frames_in_time 9
```
The first two arguments are required. The first one is the directory to start at when recursing down through the file
structure looking for images, and the second one is the directory where the .txt files containing the file paths 
will be written. The third argument is optional and indicates the filter used when determining which sub-directories 
located in ```../../data/ILSVRC2015/Data/VID/train/``` and containing video frames should be included in the .txt files. 
The last argument determines the number of consecutive frames that will be used as a single input, which also determines 
how many .txt files are written out. It is possible to just write all the frames in order on a single .txt file, 
and read in 9 consecutive frames in at a time from it, but there may be some drastic changes from one video to the next within a single input in that case. Here, we make sure that every single input contains only video frames from one video.   

## Making Nonshared Features
Since we want to model V1 simple cells, each neuron needs to have its own feature (i.e. no shared weights between neurons), and the feature needs to span the entire input image. A reasonable thing to do is to take the features learned in the [lca_dictionary_learning_shared](https://github.com/MichaelTeti/NEMO/tree/main/experiments/lca_dictionary_learning_shared) and replicate those. To do this, we use the [complex2simple.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/lca_scripts/complex2simple.py) script. This script will read in each weight file, perform this replication, and write a new weight file.   

```
python complex2simple.py --help
usage: complex2simple.py [-h] [--stride_x STRIDE_X] [--stride_y STRIDE_Y]
                         [--weight_file_key WEIGHT_FILE_KEY]
                         [--save_dir SAVE_DIR]
                         [--n_features_keep N_FEATURES_KEEP]
                         [--act_fpath ACT_FPATH] [--openpv_path OPENPV_PATH]
                         lca_ckpt_dir input_h input_w

positional arguments:
  lca_ckpt_dir          The path to the LCA checkpoint directory where the
                        weights are.
  input_h               Height of the input video frames / images.
  input_w               Width of the input video frames / images.

optional arguments:
  -h, --help            show this help message and exit
  --stride_x STRIDE_X   Stride of the original patches inside the input in the
                        x dim.
  --stride_y STRIDE_Y   Stride of the original patches inside the input in the
                        y dim.
  --weight_file_key WEIGHT_FILE_KEY
                        A key to help select out the desired weight files in
                        the ckpt_dir.
  --save_dir SAVE_DIR   The directory to save the outputs of this script in.
  --n_features_keep N_FEATURES_KEEP
                        How many features to keep.
  --act_fpath ACT_FPATH
                        Path to the <model_layer>_A.pvp file in the ckpt_dir
                        (only needed if n_features_keep is specified).
  --openpv_path OPENPV_PATH
                        Path to the OpenPv/mlab/util directory.
```

There are three required arguments. The first, ```lca_ckpt_dir``` indicates the path to the dir containing the weight files. The ```input_h``` and ```input_w``` arguements specify the height and width of the input video frames, respectively. The following image is an example of one of the convolutional features being replicated across spatial dimensions.   
![simple_grid.png](https://github.com/MichaelTeti/NEMO/blob/main/experiments/lca_dictionary_learning_nonshared/figures/feature25.gif)

## Running the Model
The model parameters are described in the [learn_imagenet_dict.lua](https://github.com/MichaelTeti/NEMO/blob/main/experiments/lca_dictionary_learning_nonshared/learn_imagenet_dict.lua) script. 

# Analysis
