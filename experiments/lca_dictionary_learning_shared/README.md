Here, we use LCA to learn a dictionary of features with convolutional sparse coding. 

# Training 
First, it is necessary to create .txt files with the file paths of the video frames the LCA model will be 
trained on in order. To do this, we use the 
[make_image_fpath_list.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/lca_scripts/make_image_fpath_list.py) 
script. First, move to the [lca_scripts](https://github.com/MichaelTeti/NEMO/tree/main/scripts/lca_scripts) directory (```cd ../../scripts/lca_scripts/```), then use the following command:
```
python3 make_image_fpath_list.py \
    ../../data/ILSVRC2015/Data/VID/train/ \
    ../../experiments/lca_dictionary_learning_shared/ \
    --key _resized \
    --n_frames_in_time 9
```
The first two arguments are required. The first one is the directory to start at when recursing down through the file
structure looking for images, and the second one is the directory where the .txt files containing the file paths 
will be saved. The third argument is optional and indicates the filter used when determining which sub-directories 
located in ```../../data/ILSVRC2015/Data/VID/train/``` and containing video frames should be included in the .txt files. 
The last argument determines the number of consecutive frames that will be used as a single input, which also determines 
how many .txt files are written out. It is possible to just write all the frames in order on a single .txt file, 
and read in 9 consecutive frames in at a time from it, but there may be some drastic changes from one video to the next within a single input in that case. Here, we make sure that every single input contains only video frames from one video.
