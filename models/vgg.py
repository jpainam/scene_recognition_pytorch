from torch import nn
import torchvision
from .block_model import ClassBlock, weights_init_kaiming
import torch


class VGGNet(nn.Module):
    def __init__(self, num_classes, pretrained=True,
                 with_attribute=False,
                 with_reweighting=False,
                 num_attrs=0,
                 num_features=2208):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        self.num_attrs = num_attrs
        self.with_reweighting = with_reweighting
        self.with_attribute = with_attribute
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

        if self.with_attribute:
            for a in range(self.num_attrs):
                self.__setattr__("attr_%d" % a,  nn.Sequential(
                                        nn.Linear(self.num_features, 128),
                                        nn.BatchNorm1d(128),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(128, 1),
                                        ))

    def forward(self, x, attrs=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.with_attribute:
            feat_attrs = [self.__getattr__("attr_%d" % a)(x) for a in range(self.num_attrs)]
            pred_attrs = torch.cat([torch.sigmoid(p) for p in feat_attrs], dim=1)
            feat_attrs = torch.cat(feat_attrs, dim=1)
            # [B x num_attrs]
            if self.with_reweighting:
                feat_attrs = self.reweigthing(feat_attrs, attrs)
                feat_attrs = feat_attrs.repeat(10, 1)
                x = torch.cat((x, feat_attrs), dim=-1)

        pred_id = self.classifier(x)
        if self.with_attribute:
            return pred_id, pred_attrs
        return pred_id