# NEMO (Neural Encoding Models for Ophys Data)
# Create an Anaconda Environment
Begginning with anaconda already installed, create an environment for this repo 
```
conda create -n NEMO python=3.6
```
Next, activate the newly-created conda environment
```
conda activate NEMO
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
2. Download and extract the neurodata (directions [here](https://github.com/MichaelTeti/NEMO/tree/main/scripts/allensdk_scripts))
