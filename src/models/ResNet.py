import torch.nn as nn


class Residualblock(nn.Module):
    def __init__(
            self, input, hidden, id=None, stride=1):
        super(Residualblock, self).__init__()
        self.ratio = 4

        # First Convolution Layer
        self.conv1 = nn.Conv2d(input, hidden, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)

        # Second Convolution Layer
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3,
                               stride=stride, padding=1, bias=False,)
        self.bn2 = nn.BatchNorm2d(hidden)

        # Third Convolution Layer
        self.conv3 = nn.Conv2d(hidden, hidden * self.ratio,
                               kernel_size=1, stride=1, padding=0, bias=False,)
        self.bn3 = nn.BatchNorm2d(hidden * self.ratio)

        self.relu = nn.ReLU()
        self.id = id
        self.stride = stride

    def forward(self, blayer):
        identity = blayer.clone()

        # First Convolution Layer
        blayer = self.conv1(blayer)
        blayer = self.bn1(blayer)
        blayer = self.relu(blayer)

        # Second Convolution Layer
        blayer = self.conv2(blayer)
        blayer = self.bn2(blayer)
        blayer = self.relu(blayer)

        # Third Convolution Layer
        blayer = self.conv3(blayer)
        blayer = self.bn3(blayer)

        if self.id is not None:
            identity = self.id(identity)

        blayer += identity
        blayer = self.relu(blayer)

        return blayer


class ResNet(nn.Module):
    def __init__(self, Residualblock, layers, channels, num_classes):
        super(ResNet, self).__init__()
        self.input = 64
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._reslayer(
            Residualblock, layers[0], output=64, stride=1)
        self.layer2 = self._reslayer(
            Residualblock, layers[1], output=128, stride=2)
        self.layer3 = self._reslayer(
            Residualblock, layers[2], output=256, stride=2)
        self.layer4 = self._reslayer(
            Residualblock, layers[3], output=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _reslayer(self, Residualblock, blocknum, output, stride):
        layers = []
        if stride != 1 or self.input != output * 4:
            id = nn.Sequential(nn.Conv2d(self.input, output * 4, kernel_size=1,
                               stride=stride, bias=False), nn.BatchNorm2d(output * 4))

        layers.append(Residualblock(self.input, output, id, stride))
        self.input = output * Residualblock(self.input, output, stride).ratio
        for _ in range(1, blocknum):
            layers.append(Residualblock(self.input, output))
        return nn.Sequential(*layers)

    def forward(self, blayer):
        blayer = self.conv1(blayer)
        blayer = self.bn1(blayer)
        blayer = self.relu(blayer)
        blayer = self.maxpool(blayer)

        blayer = self.layer1(blayer)
        blayer = self.layer2(blayer)
        blayer = self.layer3(blayer)
        blayer = self.layer4(blayer)

        blayer = self.avgpool(blayer)
        blayer = blayer.view(blayer.size(0), -1)
        blayer = self.fc(blayer)

        return blayer


def ResNet50(img_channel=3, num_classes=20):
    return ResNet(Residualblock, [3, 4, 6, 3], img_channel, num_classes)
