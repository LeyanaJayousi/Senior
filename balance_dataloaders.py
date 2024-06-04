import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

import numpy as np
import os
import src.paths as paths
from utils import *
from torch.utils import data
from torchvision import models
from collections import defaultdict
from configparser import ConfigParser
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.models.segmentation.deeplabv3 import DeepLabHead



def create_balanced_dataloaders(dataset_, train_ratio):
    """
    dataset: torch.utils.data.dataset.Subset
    """
    dataset_size = len(dataset_)

    train_count = int(dataset_size * train_ratio)
    val_count = dataset_size - train_count

    train_dataset, valid_dataset = random_split(
        dataset_, [train_count, val_count])

    y_train_indices = train_dataset.indices

    y_train = [dataset_.targets[i] for i in y_train_indices]

    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(
        samples_weight.to(torch.double), len(samples_weight))

    train_dataloader = DataLoader(train_dataset, batch_size=5, sampler=sampler)
    valid_dataloader = DataLoader(valid_dataset, batch_size=5)

    return train_dataloader, valid_dataloader


def count_occurrences(dataloader):
    class_counts = defaultdict(int)

    for _, labels in dataloader:
        for label in labels.flatten().tolist():
            class_counts[label] += 1

    for label, count in class_counts.items():
        print(f"Class {label}: {count} images")


if __name__ == '__main__':
    
    config = ConfigParser()
    config.read("config.ini")

    train_ratio = config.getfloat("Preprocessing", "train_ratio")
    meanstr = config.get("transformsVal", "mean")
    stdstr = config.get("transformsVal", "std")

    meant = [float(val) for val in meanstr.split(',')]
    stdt = [float(val) for val in stdstr.split(',')]

    transforms = T.Compose([
        T.RandomResizedCrop(size=225),
        T.RandomRotation(degrees=15),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(meant, stdt)
    ])

    mydataset = ImageFolder(root=paths.train_new, transform=transforms)

    train_dataloader, valid_dataloader = create_balanced_dataloaders(
        mydataset, train_ratio)

    # Verif.
    count_occurrences(train_dataloader)

    #
    dataloaders = {'training': train_dataloader,
                   'validation': valid_dataloader}
    dataset_sizes = {'training': len(
        train_dataloader.dataset), 'validation': len(valid_dataloader.dataset)}

    torch.save(train_dataloader, os.path.join('dataloaders', 'train.pth'))
    torch.save(valid_dataloader, os.path.join('dataloaders', 'test.pth'))
