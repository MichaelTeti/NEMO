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
    --manifest_save_dir ../../../BrainObservatoryData \
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
