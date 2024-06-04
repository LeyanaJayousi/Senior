import os
import torch
import torch.nn as nn
import torch.optim as optim
from configparser import ConfigParser


from src.models.ResNet_scratch import *
from src.models.ResNet import *


if __name__ == '__main__':
    config = ConfigParser()
    config.read("config.ini")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_dataloader_path = os.path.join("dataloaders", "train.pth")
    valid_dataloader_path = os.path.join("dataloaders", "test.pth")

    # Load the data loaders from the files
    train_dataloader = torch.load(train_dataloader_path)
    valid_dataloader = torch.load(valid_dataloader_path)

    # Create the dictionary with the data loaders
    dataloaders = {
        'training': train_dataloader,
        'validation': valid_dataloader
    }
    model = ResNet50(img_channel= config.getint("ResNet50", "image_channel"), num_classes = config.getint("ResNet50", "num_classes"))
    #Train(model = model,
          #criterion=nn.CrossEntropyLoss(),
          #optimizer=optim.Adam(model.parameters(), lr = config.getfloat("Training", "learning_rate"), 
                               #weight_decay=config.getfloat("Training", "weight_decay")),
          #num_epochs=config.getint("Training", "num_epochs"),
          #batch_size=config.getint("Training", "batch_size"),
          #dataloaders=dataloaders,
          #out_path=os.path.join(
                      #os.path.dirname(__file__), "checkpoints", "ResNetScratchAdam.pt"),
          #csv_path=os.path.join(
                    #os.path.dirname(__file__), "logs", "resnetscratchadam.csv"))
          
    # Using SGD optimizer
    Train(model = model,
          criterion=nn.CrossEntropyLoss(),
          optimizer=optim.SGD(model.parameters(), lr = config.getfloat("Training2", "learning_rate"), 
                               momentum =config.getfloat("Training2", "momentum")),
          num_epochs=config.getint("Training2", "num_epochs"),
          batch_size=config.getint("Training2", "batch_size"),
          dataloaders=dataloaders,
          model_out_path=os.path.join(
                      os.path.dirname(__file__), "checkpoints", "ResNetScratchsgd.pt"),
          csv_out_path=os.path.join(
                    os.path.dirname(__file__), "logs", "resnetscratchsgd.csv"))