#!/bin/bash

# Check if the required commands are installed
if ! command -v kaggle &> /dev/null; then
    echo "kaggle CLI could not be found, please install it first (pip install kaggle)."
    exit 1
fi

if ! command -v unzip &> /dev/null; then
    echo "unzip command could not be found, please install it first."
    exit 1
fi

KAGGLE_DATASET="meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
DOWNLOAD_DIR="data"

# Create the download directory if it does not exist
mkdir -p $DOWNLOAD_DIR

# Download the dataset
echo "Downloading dataset from Kaggle..."
kaggle datasets download -d $KAGGLE_DATASET -p $DOWNLOAD_DIR

# Find the downloaded zip file (assuming there's only one zip file)
ZIP_FILE=$(find $DOWNLOAD_DIR -name "*.zip")

if [ -z "$ZIP_FILE" ]; then
    echo "No zip file found in the download directory."
    exit 1
fi

# Unzip file
echo "Unzipping the file..."
unzip $ZIP_FILE -d $DOWNLOAD_DIR

# Remove zip
echo "Cleaning up..."
rm $ZIP_FILE

echo "Download and extraction complete."