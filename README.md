# NEMO (Neural Encoding Models for Ophys Data)
## Setup
### Prerequisites
- Required
  * [Anaconda](https://docs.anaconda.com/anaconda/install/)
- Optional
  * Cuda Toolkit>=10.2
  * Cudnn>=7.6.5

### Create an Anaconda Environment
Begginning with anaconda already installed, create an environment for this repo 
```
conda create -n nemo python=3.7
```
Next, activate the newly-created conda environment
```
conda activate nemo
```
Finally, install GNU Octave, Lua, JupyterLab, and Google-Protobuf in this environment (all other libraries will be installed 
to this environment below when running setup.py).
```
conda install -c conda-forge octave lua jupyterlab protobuf
```

### Cloning the Repository and Installing Python Packages
First, clone this repository (https shown) and run setup.py.
```
git clone git@github.com:MichaelTeti/NEMO.git &&
cd NEMO &&
python -m pip install --editable .
```

### Install PetaVision
Install PetaVision with the [install_openpv.sh](https://github.com/MichaelTeti/NEMO/blob/main/scripts/install_openpv.sh) script. 
```
bash scripts/install_openpv.sh <dir-to-install-in>
```
The script takes one positional argument, which is the directory you wish to install in (make sure you have permissions in this dir). You may need to first install some of PetaVision's dependencies (listed [here](https://github.com/PetaVision/OpenPV)), which can be installed via most Linux-based package managers.

### Downloading and Processing the Data
1. Download and pre-process the ImageNet video dataset (directions [here](https://github.com/MichaelTeti/NEMO/tree/main/scripts/image_scripts))
2. Download and extract the neurodata and stimuli (directions [here](https://github.com/MichaelTeti/NEMO/tree/main/scripts/allensdk_scripts))  
  
## Experiments
  1. [LCA Convolutional Dictionary Learning on Video](https://github.com/MichaelTeti/NEMO/tree/main/experiments/lca_dictionary_learning_shared)
  2. [LCA Simple Cell Dictionary Learning on Video](https://github.com/MichaelTeti/NEMO/tree/main/experiments/lca_dictionary_learning_nonshared)
