The scripts here are designed to download ophys experiment data given desired criteria, 
extract the data into a more friendly format, and plot the extracted data. Described below
are the steps used to obtain allensdk data in these experiments, as well as the general usage
of the different scripts. In each of the examples below, the arguments are the ones used in the 
current experiment. For the full list of arguments for each script, you can use the command
```python3 script.py --help```.
  
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
[download_nwb_files.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/download_nwb_files.py) script. The final argument is the folder where we will save all the extracted data. The last three arguments specify what should be saved. For example, removing the ```--save_rfs``` argument would cause the script to not save the receptive fields. By running this, the different desired stimuli templates will be saved in a sub-directory called Stimuli in the directory given by the third argument of the command. Similarly the eye tracking coordinates, traces, and running speed data will be saved in different sub-directories in the ExtractedData directory.

# Compute Trial Averages
The extracted fluorescence traces saved from the script above will be saved in a format with all 10 repeat trials for the given stimulus. Often, trial-averaged traces over the repeats are computed. The [get_trial_averaged_responses.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/get_trial_averaged_responses.py) script will compute these trial-averaged traces such that there will be only one trace per stimulus frame that represents the average trace across all repeats of that stimulus frame. Here we use the following command to compute these trial-averaged responses given the path to the sub-directory containing the non-trial-averaged responses:
```
python3 get_trial_averaged_responses.py \
    ../../data/BrainObservatoryData/ExtractedData/Traces/ \
    ../../data/BrainObservatoryData/ExtractedData/TrialAveragedTraces/ \
    --stimuli natural_movie_one natural_movie_three \
    --session_types three_session_A
```
The first two arguments are required, and they refer to the directory where the non-trial-averaged traces were saved and the sub-directory where you want to save the trial-averaged traces, respectively. The ```--stimuli``` argument specifies which stimuli you want to get trial-averaged traces for (assuming that it exists in the Traces sub-directory where the non-trial-averaged traces are). Here, we are using the natural_movie_one and natural_movie_three stimuli. The ```--session_types``` argument specifies the specific session types to consider when computing trial averaged traces (a depiction of the session types and how they relate to an experiment can be found [here](https://allensdk.readthedocs.io/en/latest/brain_observatory.html)). In this experiment, we use only session A because both natural movie one and natural movie three were both presented in that session (natural movie one was presented in multiple sessions, but we chose to use only the traces obtained from natural movie one in the same session as natural movie three). 

# Plotting the Data
To plot the data, there are two scripts:
* The [plot_trial_avgs_image.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/plot_trial_avgs_image.py) script is intended to be used to plot trial-averaged traces in a heatmap-style image for viewing.  
* The [plot_non_trial_avgs_line.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/plot_non_trial_avgs_line.py) script can be used to plot individual line plots for each cell/stimulus combo to visualize the traces over all repeats of the stimulus.

## Plotting Trial-Averaged Traces
To plot trial-averaged traces, we use the command
```
python3 plot_trial_avgs_image.py \
    ../../data/BrainObservatoryData/ExtractedData/TrialAveragedTraces/natural_movie_one/three_session_A/ \
    ../../data/BrainObservatoryData/ExtractedData/TrialAvgImages/natural-movie-one_three-session-A.png \
    --plot_title "Natural Movie One Normalized Fluorescence Traces"
```
which will look in the directory given by the first argument for .txt files containing the trial-averaged traces, make a plot with optional plot title given by ```--plot_title```, and save the image with the filename given by the second argument. For example, plotting the trial-averaged traces for our V1 layer IV excitatory cells to the natural movie one stimulus produced the following plot.
![](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/figures/natural-movie-one_three-session-A.png)

## Plotting Traces Across All Trials / Repeats
To plot non-trial-averaged fluorescence traces, we use the following command 
```
python3 plot_non_trial_avgs_line.py \
    ../../data/BrainObservatoryData/ExtractedData/Traces/natural_movie_one/three_session_A \
    ../../data/BrainObservatoryData/ExtractedData/NonTrialAvgTracePlots/natural_movie_one/three_session_A/ \
    --ylabel "Normalized Fluorescence Traces (95% CI shaded)"
```
where the first argument is the directory to the non-averaged traces, the second argument is the directory to save the trial-averaged plots for each cell, and the third argument is the ylabel for the plot. The reason for the third argument is because the script may be reused to plot running speed for example, and it would allow you to change the ylabel to represent that. An example plot for a given cell on the natural movie one stimulus is shown below.
![](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/figures/cellID_539774173.png)

To make sure our data extraction / processing has been done correctly, we can compare our plots vs those available on the Allen Institute's website. 
![](https://github.com/MichaelTeti/NEMO/blob/main/scripts/allensdk_scripts/figures/trace_vs_allen.png)
On the left, our plot for a given cell during the natural movie one stimulus is plotted. On the right is a plot from the Allen Institute's website for the same cell and stimulus. The right figure has repeats in red and the average as the outer blue ring. Frame 0 is at the top, and the frame numbers increase as you move clockwise around the circle. 

# Resizing the Stimuli
The monitor used to record the fluorescence traces in response to the stimuli covered [about 95 x 120 degrees of visual space](http://help.brain-map.org/download/attachments/10616846/VisualCoding_Overview.pdf?version=5&modificationDate=1538066962631&api=v2). The video frames for each natural movie stimulus are given with shape 304 x 608 pixels. Since we trained our models with video frames of shape 32 x 64, we resized them with the [resize_imgs.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/image_scripts/resize_imgs.py) script, just like with the ImageNet video frames with the command below:
```
python3 resize_imgs.py \
    ../../data/BrainObservatoryData/ExtractedData/Stimuli/ \
    32 \
    64 \
    --n_workers 20
```
