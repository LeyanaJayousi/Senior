### Traffic Sign Recognition System

## Overview
The Traffic Sign Recognition project aims to develop a deep learning model that can identify and classify traffic signs from images. This project utilizes convolutional neural networks (CNNs) using the PyTorch framework with advanced architectures such as ResNet and DeepLab to achieve high accuracy in recognizing different types of traffic signs.

## Downloading data

The dataset used for training and testing the model is the German Traffic Sign Recognition Benchmark (GTSRB).

To download the data, either

- run the following command in a terminal (no token needed) `bash ./download_dataset.sh`
- run (kaggle token needed) `python3 download_dataset.py`

For the purpose of the project, proof of concept, this project focuses on the first 20 classes of the dataset which can be extracted by executing `python3 dataAnalysis.py`scripts

## Model Architecture
The model is built using convolutional neural networks (CNNs) with the following architectures:

- ResNet (Residual Networks) from scratch (ResNet50)

- DeepLab with
    ResNet50
    ResNet101
    MobileNet

## Training
To train the model, follow these steps:

Ensure you have the dataset downloaded and extracted into the data directory.

Run `python3 balance_dataloaders.py` script to create dataloaders (80-20 split).

- For ResNet from scratch run the following script : `python3 ResNet_scratch_training.py`
Training parameters can be adjusted by modifying the `config.ini` file.

- For DeepLab models, run `python3 src.create_masks.py` to prepare the data, then run the corresponding training scripts:
For MobileNet run: `python3 src.train_mobilenet.py`
For ResNet50 run: `python3 src.train_resnet50.py`
For ResNet101 run:`python3 src.train_resnet101.py`
Training parameters can be adjusted by modifying the `src.params.ini` file.




