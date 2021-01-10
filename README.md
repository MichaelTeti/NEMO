# NEMO (Neural Encoding Models for Ophys Data)
# Create an Anaconda Environment
Begginning with anaconda already installed, create an environment for this repo 
```
conda create -n nemo python=3.8
```
Next, activate the newly-created conda environment
```
conda activate nemo
```
Finally, install octave in this environment (all other libraries will be installed 
to this environment below when running setup.py).
```
conda install -c conda-forge octave
```

# Cloning the Repository and Getting Setup
First, clone this repository (https shown) and run setup.py.
```
git clone https://github.com/MichaelTeti/NEMO.git &&
cd NEMO &&
python3 setup.py develop
```

# Downloading and Processing the Data
1. Download and pre-process the ImageNet video dataset (directions [here](https://github.com/MichaelTeti/NEMO/tree/main/scripts/image_scripts))
2. Download and extract the neurodata and stimuli (directions [here](https://github.com/MichaelTeti/NEMO/tree/main/scripts/allensdk_scripts))
