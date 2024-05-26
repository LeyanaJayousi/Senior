import torch
from utils import *
from sklearn.metrics import f1_score, roc_auc_score
from pathlib import Path
from models.DeepLabTL import *
from .losses.DiceLoss import *


def main_resnet101(data_directory, exp_directory, epochs, batch_size):

    model = createDeepLabv3_101()  # Creating deepLab model with ResNet101 architecture
    model.train()
    # data_directory = '/content/data_dir2'

    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Define the loss function (Mean Squared Error Loss)
    criterion = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4)  # Using Adam optimizer

    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    dataloaders = get_dataloader_single_folder(  # Getting dataloaders for train and test
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    torch.save(model, exp_directory / 'weights.pt')


def main_second_resnet101(data_directory, exp_directory, epochs, batch_size):

    model = createDeepLabv3_101()  # Creating DeepLab model with ResNet101 architecture
    model.train()
    data_directory = '/content/data_dir2'

    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Define the loss function (Mean Squared Error Loss)
    criterion = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01)  # Using Adam optimizer

    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    dataloaders = get_dataloader_single_folder(  # Getting dataloaders for train and test
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    torch.save(model, exp_directory / 'weights.pt')


def main_third_resnet101(data_directory, exp_directory, epochs, batch_size):

    model = createDeepLabv3_101()  # Creating DeepLab model with ResNet101 architecture
    model.train()
    data_directory = '/content/data_dir2'

    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    criterion = DiceLoss()  # Defining the loss function (Dice loss function)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01)  # Using Adam optimizer

    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    dataloaders = get_dataloader_single_folder(  # Getting dataloaders for train and test
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    torch.save(model, exp_directory / 'weights.pt')
