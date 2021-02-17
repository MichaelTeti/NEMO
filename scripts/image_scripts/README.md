These scripts are used to download and process the ImageNet video frames for the experiments. Although some of the scripts ended up not being used in the experiments, they are all still functional. The below process outlines the steps used in the experiments for downloading the videos and preprocessing them.

# Download the Dataset
To download the dataset, simply navigate to ./scripts/imagenet_scripts/ and run the [download_imagenet_vid.sh](https://github.com/MichaelTeti/NEMO/blob/main/scripts/image_scripts/download_imagenet_vid.sh) script. This script will make a data folder in the NEMO root directory and then download and extract the dataset in that directory. 
```
cd scripts/image_scripts &&
bash download_imagenet_vid.sh
```

# Resize the Images
After downloading the images, we first resize them to the appropriate size (32 x 64). To do this, we use the [preprocess_imgs.py](https://github.com/MichaelTeti/NEMO/blob/main/scripts/image_scripts/preprocess_imgs.py) script. The command is as follows:
```
python preprocess_imgs.py \
    ../../data/ILSVRC2015/Data/VID/train/ \
    resize \
    --resize_h 32 \
    --resize_ w 64 \
    --n_workers 12 \
    --aspect_ratio_tol 0.26
```
