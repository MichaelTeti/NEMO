These scripts are used to download and process the ImageNet video frames for the experiments. Although some of the scripts ended up not being used in the experiments, they are all still functional. The below process outlines the steps used in the experiments for downloading the videos and preprocessing them.

# Download the Dataset
To download the dataset, simply run the [download_imagenet_vid.sh](https://github.com/MichaelTeti/NEMO/blob/main/scripts/imagenet_scripts/download_imagenet_vid.sh) script. This script will make a data folder in the NEMO root directory and then download and extract the dataset in that directory. 

# Resize the Images
After downloading the images, we first resize them to the appropriate size (64 x 128). To do this, we use the [resize_imgs.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/imagenet_scripts/resize_imgs.py) script. The command is as follows:
```
python3 resize_imgs.py \
  ../../data/ILSVRC2015/Data/VID/train/ \
  64 \
  128 \
  --n_workers 12 \
  --aspect_ratio_tol 0.26
```
where the first argument needs to point to the directory with the video frames that are going to be resized, the second and third arguments are the desired height and width, and the fourth argument is the number of workers to use to split up the resizing and saving.
