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

# Smooth the Images
Studies in humans indicates that there is a [rough cutoff for the high frequencies we can perceive due to processing in the retina](https://www.mitpressjournals.org/doi/pdfplus/10.1162/neco.1992.4.2.196?casa_token=95yx6REIObMAAAAA:dN1g1iQV-yId9OOrdNIDCMG8nE1hYBFHR-TFuWJLN4f0lqnvIedoyoIbwr-FAGpRdbWZ_LUUITE). We can loosely approximate this by applying a smoothing filter to the video frames. Here, we use the [OpenCV bilateral filter](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed) because it will keep the edges relatively sharp, unlike many other smoothing filters, while removing some of the high frequency noise. We use the [smooth_imgs.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/imagenet_scripts/smooth_imgs.py) script to do this. 
```
python3 smooth_imgs.py \
    ../../data/ILSVRC2015/Data/VID/train/ \
    --key _resized \
    --n_workers 12 \
    --neighborhood 5 \
    --sigma_color 75 \
    --sigma_space 50
```
Here, the first argument is the path to the data directory like in the previous script. The second argument ```--key``` is also available in all of these scripts, and it allows the user to select only certain sub-directories to perform the operations on. For example, the video frames for a single video would be in ```../../data/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00111001/```, and the smoothing script will read these in and write all of the smoothed video frames in ```../../data/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00111001_smoothed/``` (all other scripts will do the same but with a different word describing their operation). The inclusion of the ```--key``` argument in the command above ensures that only the video frames that were resized are then smoothed. The last three arguments have to do with the opencv bilateralFilter function. The neighborhood function describes the diameter of the window around a given pixel to use when smoothing. The ```--sigma_color``` and ```--sigma_space``` arguments indicate the difference in color and space, respectively, two pixels in a neighborhood have to be to be combined in the smoothing operation for the given pixel that is being smoothed. Here, we use the default arguments provided in the opencv examples for the bilateralFilter function, which seemed to filter out some of the high frequency noise in these video frames while not smoothing too much.
