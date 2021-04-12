from torch import nn
import torchvision
from .block_model import ClassBlock


class DenseNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, num_features=2208, **kwargs):
        super(DenseNet, self).__init__()
        self.num_classes = num_classes
        model_ft = torchvision.models.densenet161(pretrained=pretrained)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.features = model_ft.features
        self.num_features = num_features
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(self.num_features, 1024),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Linear(1024, self.num_classes),
                                        )

    def forward(self, x, attrs=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)