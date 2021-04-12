import torch
from torch import nn


class ResNext(nn.Module):
    def __init__(self, num_classes,
                 with_attribute=False,
                 with_reweighting=False,
                 num_features=2048,
                 num_attrs=0, **kwargs):
        super(ResNext, self).__init__()
        self.num_classes = num_classes
        self.with_attribute = with_attribute
        self.with_reweighting = with_reweighting
        self.num_attrs = num_attrs
        self.num_features = num_features

        #self.features = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
        self.features = torch.hub.load('facebookresearch/WSL-Images',
                                       'resnext101_32x48d_wsl')
        self.features.fc = nn.Sequential()
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(2048, 1024),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Linear(1024, self.num_classes),

                                        )

        if self.with_attribute:
            for a in range(self.num_attrs):
                self.__setattr__("attr_%d" % a,  nn.Sequential(
                                        nn.Linear(self.num_features, 512),
                                        nn.BatchNorm1d(512),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(512, 1),
                                        ))

    def forward(self, x, attrs=None):
        x = self.features(x)

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
