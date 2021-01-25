# NEMO (Neural Encoding Models for Ophys Data)
## Setup
### System Specifications
Experiments were run on a system with the following specifications:
* CentOS 7.8 / Ubuntu 18.04
* NVIDIA Cuda Toolkit 10.2
* NVIDIA CUDNN 7
* Python 3.7
* Power9 Processors
* Tesla V100 GPUs

### Create an Anaconda Environment
Begginning with anaconda already installed, create an environment for this repo 
```
conda create -n nemo python=3.7
```
Next, activate the newly-created conda environment
```
conda activate nemo
```
Finally, install GNU Octave, Lua, and JupyterLab in this environment (all other libraries will be installed 
to this environment below when running setup.py).
```
conda install -c conda-forge octave lua jupyterlab
```

### Cloning the Repository and Installing Python Packages
First, clone this repository (https shown) and run setup.py.
```
git clone https://github.com/MichaelTeti/NEMO.git &&
cd NEMO &&
python3 setup.py develop
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
