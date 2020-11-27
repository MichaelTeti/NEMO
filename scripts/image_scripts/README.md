These scripts are used to download and process the ImageNet video frames for the experiments. Although some of the scripts ended up not being used in the experiments, they are all still functional. The below process outlines the steps used in the experiments for downloading the videos and preprocessing them.

# Download the Dataset
To download the dataset, simply navigate to ./scripts/imagenet_scripts/ and run the [download_imagenet_vid.sh](https://github.com/MichaelTeti/NEMO/blob/main/scripts/image_scripts/download_imagenet_vid.sh) script. This script will make a data folder in the NEMO root directory and then download and extract the dataset in that directory. 
```
cd scripts/imagenet_scripts &&
bash download_imagenet_vid.sh
```

# Resize the Images
After downloading the images, we first resize them to the appropriate size (32 x 64). To do this, we use the [resize_imgs.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/image_scripts/resize_imgs.py) script. The command is as follows:
```
python3 resize_imgs.py \
    ../../data/ILSVRC2015/Data/VID/train/ \
    32 \
    64 \
    --n_workers 12 \
    --aspect_ratio_tol 0.26
```

This script has the following arguments 
```
usage: resize_imgs.py [-h] [--n_workers N_WORKERS] [--key KEY]
                      [--aspect_ratio_tol ASPECT_RATIO_TOL]
                      data_dir_parent desired_height desired_width

positional arguments:
  data_dir_parent       The parent directory with video frames and/or subdirs
                        with video frames.
  desired_height
  desired_width

optional arguments:
  -h, --help            show this help message and exit
  --n_workers N_WORKERS
  --key KEY             If key is specified and not in path of a frame, that
                        frame will not be used.
  --aspect_ratio_tol ASPECT_RATIO_TOL
                        If actual aspect ratio is > aspect_ratio_tol, frames
                        will not be resized.
```

where the first argument is required and needs to point to the directory with the video frames that are going to be resized, the second and third arguments are also required and are the desired height and width, and the fourth argument is the number of workers to use to split up the resizing and saving (should be no more than about nprocs / 2). The last argument is used so that the script will ignore images with an aspect ratio more than ```aspect_ratio_tol``` different compared to the aspect ratio of the desired dimensions given. The ```--key``` argument is not used here, but is helpful if you want to do multiple operations on the original video frames via the scripts in this directory (e.g. resizing, then cropping, then whitening, etc.). For example, the [resize_imgs.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/image_scripts/resize_imgs.py) script here will recurse through the ```data_dir_parent```, and when it finds video frames it will resize them and then save them in a subdirectory with the same name as the original but with "\_resized" at the end. If you then wanted to whiten the resized images, you would run the 
[whiten_imgs.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/image_scripts/whiten_imgs.py) script with ```--key _resized``` as a command-line argument to avoid having to whiten all of the video frames from the original, unresized directories.
