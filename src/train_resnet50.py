from configparser import ConfigParser

import torch
import torch.nn as nn
import warnings

from models.DeepLabResNet50 import *
from paths import *


if __name__ == '__main__':

    config = ConfigParser()
    config.read(os.path.join("src", "params.ini"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main_resnet50(data_directory=data_directory,
                  exp_directory=os.path.join(
                      os.path.dirname(__file__), "logs"),
                  num_epochs=config.getint("ResNet50", "num_epochs"),
                  batch_size=config.getint("ResNet50", "batch_size"),
                  loss_fn=nn.MSELoss(reduction="mean"),
                  lr=config.getfloat("ResNet50", "lr"),
                  checkpoint_path=os.path.join(
                      os.path.dirname(__file__), "checkpoints", "resnet50.pt"),
                  device=device)
