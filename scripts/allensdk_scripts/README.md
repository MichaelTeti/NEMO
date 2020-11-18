The scripts here are designed to download ophys experiment data given desired criteria, 
extract the data into a more friendly format, and plot the extracted data. Described below
are the steps used to obtain allensdk data in these experiments, as well as the general usage
of the different scripts.  
  
# Download the Data  
First, download the experiment containers based on your criteria with the 
[download_nwb_files.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/download_nwb_files.py)
script. This script takes in a number of arguments, many of which are optional and allow the user to filter out 
containers which do not contain certain desired criteria, such as imaging depth, targeted structure, cre lines, etc. 
For this experiment the following command and arguments were used:  
```
python3 download_nwb_files.py \
    --manifest_save_dir ../../data/BrainObservatoryData \
    --n_workers 4 \
    --cre_lines "Rorb-IRES2-Cre" "Scnn1a-Tg3-Cre" "Nr5a1-Cre" \
    --reporter_lines "Ai93(TITL-GCaMP6f)" \
    --targeted_structures "VISp"
```  
The ```--manifest_save_dir``` argument specifies where the manifest file will be saved. The manifest file is basically
a file that contains the metadata for all the experiments. The ```--n_workers``` argument in this script specifies how
many experiment containers will be downloaded at once. The ```--cre_lines``` and ```--reporter_lines``` arguments take in the desired cre lines you
want based on what cells you are trying to isolate. Here, we use the three cre lines above with the one reporter line because they 
isolate mostly excitatory cells in layer IV. The characteristics of the different cre lines and reporter lines can be found
[here](http://observatory.brain-map.org/visualcoding/transgenic). Finally, we use the ```--targeted_structures``` argument
here to indicate that we only want cells in the primary visual cortex, but the different structures available can be found 
[here](http://observatory.brain-map.org/visualcoding).

# Extract the Data
Now that the experiment containers were downloaded, we want to extract the relevant data inside of them using the 
[extract_neurodata.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/extract_neurodata.py) script. 
This script will perform three main functions:
* Extract and save the desired stimuli  
* Extract and save each cell's fluorescence traces for each video frame of the stimulus, as well as the corresponding eye 
tracking coordinates and running speed. 
* Extract and save each cell's on and off receptive fields obtained using the locally sparse noise stimulus. 

At this time, these functions are only performed w.r.t. the natural movie stimuli because that is what we use in these
experiments. To run this script for our experiments, we use the command:  
```
python3 extract_neurodata.py \
   ../../data/BrainObservatoryData/manifest.json \
   ../../data/BrainObservatoryData/ophys_experiment_data/ \
   ../../data/BrainObservatoryData/ExtractedData \
   --stimuli natural_movie_one natural_movie_three \
   --n_workers 16 \
   --save_traces_and_pupil_coords \
   --save_stimuli \
   --save_rfs
```
The first three lines correspond to the required arguments of the script. The first one is the path to the manifest 
file. The second is the path to the ophys data that is downloaded and created with the 
[download_nwb_files.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/download_nwb_files.py) script. The final argument is the folder where we will save all the extracted data. The last three arguments specify what should be saved. For example, removing the ```--save_rfs``` argument would cause the script to not save the receptive fields. 
