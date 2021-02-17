Here are scripts that are used in the LCA pipeline (mostly when using PetaVision to do LCA).

## complex2simple.py   
This [complex2simple.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/lca_scripts/complex2simple.py) script takes in the shared features from the convolutional LCA model and replicates each one across space to simulate simple cell features. 
The arguments are as follows:

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
