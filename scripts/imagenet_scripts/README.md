These scripts are used to download and process the ImageNet video frames for the experiments. Although some of the scripts ended up not being used in the experiments, they are all still functional. The below process outlines the steps used in the experiments for downloading the videos and preprocessing them.

# Download the Dataset
To download the dataset, simply navigate to ./scripts/imagenet_scripts/ and run the [download_imagenet_vid.sh](https://github.com/MichaelTeti/NEMO/blob/main/scripts/imagenet_scripts/download_imagenet_vid.sh) script. This script will make a data folder in the NEMO root directory and then download and extract the dataset in that directory. 
```
cd scripts/imagenet_scripts &&
bash download_imagenet_vid.sh
```

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
where the first argument needs to point to the directory with the video frames that are going to be resized, the second and third arguments are the desired height and width, and the fourth argument is the number of workers to use to split up the resizing and saving. The last argument is used so that the script will ignore images with an aspect ratio more than ```aspect_ratio_tol``` different compared to the aspect ratio of the desired dimensions given.

# Whiten the Images (Optional)
Whitening is a common step used in image processing, but it is much less common for current SOTA learning algorithms (i.e. convolutional neural networks). Here, we use ZCA whitening on the resized images with the following command.
```
python3 whiten_imgs.py \
  ../../data/ILSVRC2015/Data/VID/train/ \
  --key _resized \
  --full_svd
```
Like most of the image processing scripts in this directory, the first argument is going to be the directory containing the video frames and/or other sub-directories that each contain the video frames from a video. The second argument is also available in all of these scripts, and it allows the user to select only certain sub-directories to perform the operations on. For example, the video frames for a single video would be in ```../../data/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00111001/```, and the resizing script will read these in and write all of the resized video frames in ```../../data/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00111001_resized/``` (all other scripts will do the same but with a different word describing their operation). The inclusion of the ```--key``` argument in the command above ensures that only the video frames that were resized are then whitened. 
