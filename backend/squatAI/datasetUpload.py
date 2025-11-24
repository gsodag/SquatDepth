import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Specify dataset (replace with your dataset's identifier, e.g., 'username/dataset-name')
dataset = 'ayoobaboosalih/powerlifting-squat-dataset'

# Download dataset to a specified path
download_path = './dataset'  # Folder where dataset will be saved
api.dataset_download_files(dataset, path=download_path, unzip=True)