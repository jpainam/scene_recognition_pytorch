import torch
from torch import nn


class ResNext(nn.Module):
    def __init__(self, num_classes):
        super(ResNext, self).__init__()
        self.num_classes = num_classes
        self.features = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
        self.features.fc = nn.Sequential()
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(2048, 1024),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Linear(1024, self.num_classes),

                                        )

    def forward(self, x, attrs=None):
        x = self.features(x)
        return self.classifier(x)
