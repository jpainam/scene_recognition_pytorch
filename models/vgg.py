from torch import nn
import torchvision
from .block_model import ClassBlock, weights_init_kaiming


class VGGNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, num_features=2208):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        model_ft = torchvision.models.vgg19(pretrained=pretrained)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        model_ft.classifier = nn.Sequential()
        self.features = model_ft.features
        self.num_features = num_features
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(self.num_features, 1024),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Linear(1024, self.num_classes),
                                        )
        self.classifier.apply(weights_init_kaiming)

    def forward(self, x, attrs=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)