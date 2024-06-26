import os
import src.paths as paths
import uuid
import shutil
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
from io import StringIO
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import copy
import time
from tqdm import tqdm
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_training_results(log_file_path):
    """
    Plot results for deeplab models.
    """
    with open(log_file_path, 'r') as file:
        reader = csv.DictReader(file)
        epochs = []
        losses = []
        fone = []
        aurocs = []
        for row in reader:
            epochs.append(int(row['epoch']))
            losses.append(float(row['Train_loss']))
            fone.append(float(row['Train_f1_score']))
            aurocs.append(float(row['Train_auroc']))

    # Create plots
    fig = plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    # Plot F1 score
    plt.subplot(1, 3, 2)
    plt.plot(epochs, fone, marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.title('Training F1 score')

    # Plot AUC-ROC value
    plt.subplot(1, 3, 3)
    plt.plot(epochs, aurocs, marker='o', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('AUC-ROC value')
    plt.title('Training AUC-ROC value')

    plt.tight_layout()
    
    return fig

def save_figures(figures_folder, plot_function, file_name, *args):
    """
    Saves the given plot function output as PDF and PNG in the specified folder.
    """
    # Ensure the figures folder exists
    os.makedirs(figures_folder, exist_ok=True)

    # Check if the plot function requires arguments and call accordingly
    if args:
        fig = plot_function(*args)
    else:
        fig = plot_function()

    # Define file paths for PDF and PNG
    pdf_path = os.path.join(figures_folder, f'{file_name}.pdf')
    png_path = os.path.join(figures_folder, f'{file_name}.png')

    # Save the figure as PDF and PNG
    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png')

    plt.close(fig)


def process_data(input_path):
    """
    Process the data based on the input type (CSV file or directory).
    If it's a CSV file, filter and create a new CSV file.
    If it's a directory, extract the first 20 folders and create a new dataset.
    """
    new_dataset_path = os.path.join(paths.dataset_path, 'new_dataset')
    os.makedirs(new_dataset_path, exist_ok=True)

    if input_path.endswith('.csv'):
        # Handle CSV file
        df = pd.read_csv(input_path)
        df = df[df['ClassId'] <= 19]
        file_name = os.path.basename(input_path)
        new_csv_path = os.path.join(new_dataset_path, file_name)
        df.to_csv(new_csv_path, index=False)
        return new_csv_path
    elif os.path.isdir(input_path):
        # Handle directory
        all_folders = [folder for folder in os.listdir(
            input_path) if os.path.isdir(os.path.join(input_path, folder))]
        sorted_folders = sorted(all_folders, key=lambda x: int(x))[:20]
        new_train_path = os.path.join(new_dataset_path, 'Train')
        os.makedirs(new_train_path, exist_ok=True)
        for folder in sorted_folders:
            src = os.path.join(input_path, folder)
            dst = os.path.join(new_train_path, folder)
            shutil.copytree(src, dst)
        return new_train_path
    else:
        raise ValueError("The input path must be a CSV file or a directory.")


def evaluate_model(model, dataloader):
    """
     method to evaluate the performance and print a confusion matrix
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def plot_confusion_matrix(cm, classes):
    """
    method to print the confusion matrix
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


