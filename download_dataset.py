import os
from kaggle.api.kaggle_api_extended import KaggleApi

KAGGLE_DATASET = 'meowmeowmeowmeowmeow/gtsrb-german-traffic-sign'
DOWNLOAD_DIR = 'data'

# Create the download directory if it does not exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset
print("Downloading dataset from Kaggle...")
api.dataset_download_files(KAGGLE_DATASET, path=DOWNLOAD_DIR, unzip=True)

print("Download and extraction complete.")