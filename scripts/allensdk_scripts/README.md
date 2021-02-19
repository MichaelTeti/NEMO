The scripts here are designed to download ophys experiment data given desired criteria and 
extract the data into a more friendly format. Described below
are the steps used to obtain allensdk data in these experiments, as well as the general usage
of the different scripts.
  
# Download the Data  
First, download the experiment containers based on your criteria with the 
[download_nwb_files.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/download_nwb_files.py)
script. This script takes in a number of arguments, many of which are optional and allow the user to filter out 
experiments based on certain desired criteria, such as imaging depth, targeted structure, cre lines, etc. 
The full range of arguments for this script are the following:

```
python download_nwb_files.py --help
usage: download_nwb_files.py [-h] [--manifest_save_dir MANIFEST_SAVE_DIR]
                             [--n_workers N_WORKERS]
                             [--n_experiments N_EXPERIMENTS]
                             [--targeted_structures {VISp,VISl,VISal,VISrl,VISam,VISpm} [{VISp,VISl,VISal,VISrl,VISam,VISpm} ...]]
                             [--cre_lines {Cux2-CreERT2,Emx1-IRES-Cre,Fezf2-CreER,Nr5a1-Cre,Ntsr1-Cre_GN220,Pvalb-IRES-Cre,Rbp4-Cre_KL100,Rorb-IRES2-Cre,Scnn1a-Tg3-Cre,Slc17a7-IRES2-Cre,Sst-IRES-Cre,Tlx3-Cre_PL56,Vip-IRES-Cre} [{Cux2-CreERT2,Emx1-IRES-Cre,Fezf2-CreER,Nr5a1-Cre,Ntsr1-Cre_GN220,Pvalb-IRES-Cre,Rbp4-Cre_KL100,Rorb-IRES2-Cre,Scnn1a-Tg3-Cre,Slc17a7-IRES2-Cre,Sst-IRES-Cre,Tlx3-Cre_PL56,Vip-IRES-Cre} ...]]
                             [--reporter_lines {Ai148TIT2L-GC6f-ICL-tTA2),Ai162(TIT2L-GC6s-ICL-tTA2),Ai93(TITL-GCaMP6f),Ai93(TITL-GCaMP6f)-hyg,Ai94(TITL-GCaMP6s)} [{Ai148(TIT2L-GC6f-ICL-tTA2),Ai162(TIT2L-GC6s-ICL-tTA2),Ai93(TITL-GCaMP6f),Ai93(TITL-GCaMP6f)-hyg,Ai94(TITL-GCaMP6s} ...]]
                             [--min_imaging_depth MIN_IMAGING_DEPTH]
                             [--max_imaging_depth MAX_IMAGING_DEPTH]
                             [--session_type {three_session_A,three_session_B,three_session_C,three_session_C2} [{three_session_A,three_session_B,three_session_C,three_session_C2} ...]]

optional arguments:
  -h, --help            show this help message and exit
  --manifest_save_dir MANIFEST_SAVE_DIR
                        Where to save the manifest file. Default is in the
                        current directory.
  --n_workers N_WORKERS
                        Number of CPUs to use when downloading NWB files.
                        Default is 1.
  --n_experiments N_EXPERIMENTS
                        Number of experiments to download. Default is all of
                        them.
  --targeted_structures {VISp,VISl,VISal,VISrl,VISam,VISpm} [{VISp,VISl,VISal,VISrl,VISam,VISpm} ...]
                        Targeted structures. Default is all of them.
  --cre_lines {Cux2-CreERT2,Emx1-IRES-Cre,Fezf2-CreER,Nr5a1-Cre,Ntsr1-Cre_GN220,Pvalb-IRES-Cre,Rbp4-Cre_KL100,Rorb-IRES2-Cre,Scnn1a-Tg3-Cre,Slc17a7-IRES2-Cre,Sst-IRES-Cre,Tlx3-Cre_PL56,Vip-IRES-Cre} [{Cux2-CreERT2,Emx1-IRES-Cre,Fezf2-CreER,Nr5a1-Cre,Ntsr1-Cre_GN220,Pvalb-IRES-Cre,Rbp4-Cre_KL100,Rorb-IRES2-Cre,Scnn1a-Tg3-Cre,Slc17a7-IRES2-Cre,Sst-IRES-Cre,Tlx3-Cre_PL56,Vip-IRES-Cre} ...]
                        Desired Cre lines. Default is all of them.
  --reporter_lines {Ai148(TIT2L-GC6f-ICL-tTA2),Ai162(TIT2L-GC6s-ICL-tTA2),Ai93(TITL-GCaMP6f),Ai93(TITL-GCaMP6f)-hyg,Ai94(TITL-GCaMP6s)} [{Ai148(TIT2L-GC6f-ICL-tTA2),Ai162(TIT2L-GC6s-ICL-tTA2),Ai93(TITL-GCaMP6f),Ai93(TITL-GCaMP6f)-hyg,Ai94(TITL-GCaMP6s)} ...]
                        Desired reporter lines. Default is all of them.
  --min_imaging_depth MIN_IMAGING_DEPTH
                        Minimum desired imaging depth.
  --max_imaging_depth MAX_IMAGING_DEPTH
                        Maximium desired imaging depth.
  --session_type {three_session_A,three_session_B,three_session_C,three_session_C2} [{three_session_A,three_session_B,three_session_C,three_session_C2} ...]
                        Choose a specific session type to pull if desired.
```

For this experiment the following command was used to download experiments with V1 simple cell data:  
```
python download_nwb_files.py \
    --manifest_save_dir ../../data/AIBO/VISp/L4 \
    --n_workers 4 \
    --cre_lines \
        "Rorb-IRES2-Cre" \
        "Scnn1a-Tg3-Cre" \
        "Nr5a1-Cre" \
        "Cux2-CreERT2" \
        "Emx1-IRES-Cre" \
        "Slc17a7-IRES2-Cre" \
    --reporter_lines \
        "Ai93(TITL-GCaMP6f)" \
        "Ai94(TITL-GCaMP6s)" \
    --targeted_structures "VISp" \
    --min_imaging_depth 275 \
    --max_imaging_depth 450
```  
The ```--manifest_save_dir``` argument specifies where the manifest file will be saved. The manifest file is basically
a file that has different information about the containers and files. The ```--n_workers``` argument in this script specifies how
many experiment datasets will be downloaded in parallel. The ```--cre_lines``` and ```--reporter_lines``` arguments take in the desired cre lines you
want based on what cells you are trying to isolate. The characteristics of the different cre lines and reporter lines can be found
[here](http://help.brain-map.org/download/attachments/10616846/VisualCoding_TransgenicCharacterization.pdf?version=4&modificationDate=1538067045225&api=v2).
Essentially, we search for all cre and reporter lines that correspond to layer 4 excitatory cells.  
![](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/figures/transgenic_lines.png)
  
For the ```--targeted_structures``` argument, we use ```"VISp"```
to indicate that we only want cells in the primary visual cortex. The different structures available can be found 
[here](http://observatory.brain-map.org/visualcoding) and in the argument choices for the script. Since most of the transgenic lines we pulled out correspond
to cells in multiple cortical layers, we use the ```--min_imaging_depth``` and ```--max_imaging_depth``` arguments to specify a cortical depth range that roughly corresponds to the [depth spanned by layer 4 in the cortex](http://www.nibb.ac.jp/brish/Gallery/cortexE.html) (also mentioned in [this paper](https://sci-hub.se/10.3791/60600)).

# Extract the Data
Now that the experiment containers were downloaded, we want to extract and write the relevant data inside of them using the 
[extract_neurodata.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/extract_neurodata.py) script. 
This script will extract and write the following (as well as a few other things):
  * Stimuli templates for natural movies, natural scenes, and static gratings (working on drifting gratings)
  * each cell's df/f
  * the animal's running speed
  * the animal's eye tracking data
  * each cell's receptive field computed with locally sparse noise 

The script takes in the following arguments:

```
python extract_neurodata.py --help
usage: extract_neurodata.py [-h] [--no_stim_or_trace_data] [--no_rfs]
                            [--n_workers N_WORKERS]
                            exp_dir manifest_fpath save_dir

positional arguments:
  exp_dir               Directory containing the .nwb experiment files.
  manifest_fpath        Path to the manifest.json file.
  save_dir              Where to save all extracted data.

optional arguments:
  -h, --help            show this help message and exit
  --no_stim_or_trace_data
                        If specified, will not write out stimuli or trace
                        data.
  --no_rfs              If specified, will not write out receptive fields.
  --n_workers N_WORKERS
                        Number of datasets to write in parallel.
```

To extract the data downloaded in the previous step, we use the following command:

```
python extract_neurodata.py \
    ../../data/AIBO/VISp/L4/ophys_experiment_data/ \
    ../../data/AIBO/VISp/L4/manifest.json \
    ../../data/AIBO/VISp/L4/ExtractedData/ \
    --n_workers 20
```

The first three arguments are required. The first one is the path to the directory of .nwb dataset files which were downloaded. The second is the path to the manifest.json file. The final argument is the folder where we will write all the extracted data. Inside of this folder, the stimuli templates will be written in a subdirectory called stimuli, whereas the behavioral / trace data will be written in a subdirectory called trace_data, and the receptive fields will be written in a subdirectory called receptive_fields. The last two arguments allow you to forego writing either the receptive fields or stimuli and behavioral / df/f data. The ```--n_workers``` argument specifies how many datasets to write in parallel, and depends on your system's specifications. For each stimulus type (i.e. natural movies, natural scenes, static gratings) / cell ID combination, there will be a single .csv file written with the cell's behavioral and trace data. The stimuli frames will be warped as they were presented on the monitor to the mouse (information on this can be found [here](http://help.brain-map.org/download/attachments/10616846/VisualCoding_VisualStimuli.pdf?version=3&modificationDate=1497305590322&api=v2)), and will be written as 1920 x 1200 pixel images. Below is an example of the warping that is done to the stimuli with a frame from the natural scenes stimulus.  
   
![](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/figures/082.png)
