
from configparser import ConfigParser

import os
import torch
import torch.nn as nn
import warnings

from models.DeepLabMobileNet import *
from models.losses.DiceLoss import *
from paths import *

if __name__ == '__main__':

    config = ConfigParser()
    config.read(os.path.join("src", "params.ini"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #Using MSE loss function
    #main_mobilenet(data_directory=data_directory,
                  #exp_directory=os.path.join(
                      #os.path.dirname(__file__), "logs"),
                  #num_epochs=config.getint("deeplabmodels", "num_epochs"),
                  #batch_size=config.getint("deeplabmodels", "batch_size"),
                  #loss_fn=nn.MSELoss(reduction="mean"),
                  #lr=config.getfloat("deeplabmodels", "lr"),
                  #checkpoint_path=os.path.join(
                      #os.path.dirname(__file__), "checkpoints", "mobilenetmse.pt"),
                  #log_name=os.path.join(
                      #os.path.dirname(__file__), "logs", "mobilenetmse.csv"),
                  #device=device)
    #Using Diceloss
    main_mobilenet(data_directory=data_directory,
                  exp_directory=os.path.join(
                      os.path.dirname(__file__), "logs"),
                  num_epochs=config.getint("deeplabmodels", "num_epochs"),
                  batch_size=config.getint("deeplabmodels", "batch_size"),
                  loss_fn=DiceLoss(),
                  lr=config.getfloat("deeplabmodels", "lr"),
                  checkpoint_path=os.path.join(
                      os.path.dirname(__file__), "checkpoints", "mobilenetdice.pt"),
                  log_name=os.path.join(
                      os.path.dirname(__file__), "logs", "mobilenetdice.csv"),
                  device=device)
