import os
import paths as paths
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


def list_subfolders(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    return subfolders


def filter_and_create_csv(csv_file, fileName):
    """
    creating new csv files with ClassId < 20
    """
    df = pd.read_csv(csv_file)
    df = df[df['ClassId'] <= 19]
    new_dataset_path = os.path.join(paths.train_folder, 'new_csv')
    os.makedirs(new_dataset_path, exist_ok=True)
    new_csv_file = os.path.join(new_dataset_path, fileName)
    df.to_csv(new_csv_file, index=False)

    return new_csv_file


def get_unique_filename(prefix='model', suffix='.pt', folder='/content'):
    """
    method to print the model in a separate file
    """
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{unique_id}{suffix}"
    return os.path.join(folder, filename)


def extract_20_classes(train_folder):
    """
    extracting first 20 folders (0-19) from train folder to minimize the dataset
    """

    all_folders = [folder for folder in os.listdir(
        paths.train_folder) if os.path.isdir(os.path.join(paths.train_folder, folder))]
    sorted_folders = sorted(all_folders, key=lambda x: int(x))[:20]
    new_dataset_path = os.path.join(paths.train_folder, 'new_dataset')
    os.makedirs(new_dataset_path, exist_ok=True)
    new_train_path = os.path.join(new_dataset_path, 'Train')
    os.makedirs(new_train_path, exist_ok=True)
    for folder in sorted_folders:
        src = os.path.join(paths.train_folder, folder)
        dst = os.path.join(new_train_path, folder)
        shutil.copytree(src, dst)

    return new_train_path


def Train(model, criterion, optimizer, num_epochs, batch_size, dataloaders, out_path):
    """
    Method to train the model , with necessary parameters

    """

    highest_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_labels = []
            all_preds = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_accuracy = accuracy_score(all_labels, all_preds)
            epoch_balanced_accuracy = balanced_accuracy_score(
                all_labels, all_preds)

            print(f'{phase.capitalize()} Epoch {epoch + 1}/{num_epochs} '
                  f'Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f} '
                  f'Balanced Accuracy: {epoch_balanced_accuracy:.4f}\n')

            if phase == 'validation' and epoch_accuracy > highest_acc:
                highest_acc = epoch_accuracy

    torch.save(model.state_dict(), out_path)

    print('Highest accuracy is: {:.4f}'.format(highest_acc))
    return highest_acc


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


def capture_output(func, *args, **kwargs):
    """
    capturing outputs to plot graphs
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        func(*args, **kwargs)
        return sys.stdout.getvalue().splitlines()
    finally:
        sys.stdout = old_stdout


def create_and_copy_folders(source_train_path, source_mask_path, destination_path):
    """
    Creates 'Images' and 'Masks' directories in the specified destination path
    and copies image files from the source training and mask paths into these
    directories, renaming them sequentially.
    """
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    train_dir = os.path.join(destination_path, 'Images')
    mask_dir = os.path.join(destination_path, 'Masks')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    train_image_counter = 1
    for subdir, _, files in os.walk(source_train_path):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                shutil.copy(os.path.join(subdir, file), os.path.join(
                    train_dir, f"Image{train_image_counter}.jpg"))
                train_image_counter += 1

    mask_image_counter = 1
    for subdir, _, files in os.walk(source_mask_path):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                shutil.copy(os.path.join(subdir, file), os.path.join(
                    mask_dir, f"Image{mask_image_counter}_label.png"))
                mask_image_counter += 1


def createDeepLabv3(outputchannels=1):
    """
    Creating DeepLab model with ResNet50 architecture
    """

    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'deeplabv3_resnet50', pretrained=True)

    model.classifier = DeepLabHead(2048, outputchannels)

    model.train()
    return model


def createDeepLabv3_101(outputchannels=1):
    """
    Creating DeepLab model with ResNet101 architecture
    """

    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'deeplabv3_resnet101', pretrained=True)

    model.classifier = DeepLabHead(2048, outputchannels)

    model.train()
    return model


def createDeepLabv3_mobileNet(outputchannels=1):
    """
    Creating DeepLab model with MobileNet architecture
    """

    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'deeplabv3_mobilenet_v3_large', pretrained=True)

    model.classifier = DeepLabHead(960, outputchannels)

    model.train()
    return model


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs):
    """ Training DeepLab model"""

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == 'f1_score':

                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))
                        else:
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)

            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    return model


def plot_training_results(log_file_path):
    """
    Plot results for deeplab models
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
    plt.figure(figsize=(15, 5))

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
    plt.show()
