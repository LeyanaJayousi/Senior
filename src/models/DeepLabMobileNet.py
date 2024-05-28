import torch

from sklearn.metrics import f1_score, roc_auc_score
from pathlib import Path
from .DeepLabTL import *
from .losses.DiceLoss import *
from .train_model import *


from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def createDeepLabv3_mobileNet(outputchannels=1):
    """
    Creating DeepLab model with MobileNet architecture
    """

    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'deeplabv3_mobilenet_v3_large', pretrained=True)

    model.classifier = DeepLabHead(960, outputchannels)

    model.train()
    return model


def main_mobilenet(data_directory, exp_directory, num_epochs, batch_size, loss_fn, lr, checkpoint_path, log_name, device="cpu"):

    model = createDeepLabv3_mobileNet()  # Create the DeepLabv3 model with ResNet50 architecture
    model.train()

    exp_directory = Path(exp_directory)  # output directory

    if not exp_directory.exists():
        exp_directory.mkdir()

    # Defining the loss function (Mean Squared Error Loss)
    criterion = loss_fn  # or DiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr)  # Using Adam optimizer

    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    dataloaders = get_dataloader_single_folder(  # Getting the dataloaders for train and test
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=num_epochs,
                    checkpoint_path=checkpoint_path,
                    log_name=log_name,
                    device=device)