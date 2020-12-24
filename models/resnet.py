import torch
from torch import nn
import torchvision
from torch.nn import init

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
    }

    def __init__(self, depth, pretrained=True, num_features=0, dropout=0, norm=False,
                 num_classes=0, pool="avg", stride=1, num_attrs=0):
        super(ResNet, self).__init__()
        assert num_classes != 0, 'The number of classes must be non null'
        self.depth = depth
        self.pretrained = pretrained
        self.num_features = num_features
        self.has_embedding = num_features > 0
        self.norm = norm
        self.num_classes = num_classes

        if depth not in ResNet.__factory:
            raise KeyError(f"Unsupported resnet depth {depth} module, must be [18, 34, 50, 101]")
        self.model = ResNet.__factory[depth](pretrained=pretrained)
        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1, 1)
            self.model.layer4[0].conv2.stride = (1, 1)

        self.dropout = dropout
        out_planes = self.model.fc.in_features
        # Append a bottleneck layer
        if self.num_features > 0:
            self.feat = nn.Sequential(
                nn.Linear(out_planes, self.num_features),
                nn.BatchNorm1d(self.num_features),
            )
            self.feat.apply(weights_init_classifier)
            self.feat.apply(weights_init_kaiming)
        else:
            self.num_features = out_planes

        if dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        self.pool = pool
        if "avg" in pool:
            self.model.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        if "max" in pool:
            self.model.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))

        if "avg" in pool and "max" in pool:
            self.num_features = self.num_features * 2

        self.bn = nn.BatchNorm1d(self.num_features)
        # Classification layer
        self.classifier = nn.Linear(self.num_features, self.num_classes)
        self.classifier.apply(weights_init_classifier)

        self.num_attrs = num_attrs
        if self.num_attrs > 0:
            self.attr_classifier = nn.Linear(self.num_features, self.num_attrs)
            self.attr_classifier.apply(weights_init_classifier)


    '''def _inflate_reslayer(self, reslayer, height=0, width=0,
                          alpha_x=0, alpha_y=0, IA_idx=[], IA_channels=0):
        reslayers = []
        for i, layer2d in enumerate(reslayer):
            reslayers.append(layer2d)

            if i in IA_idx:
                IA_block = IABlock2D(in_channels=IA_channels, height=height,
                                     width=width, alpha_x=alpha_x, alpha_y=alpha_y)
                reslayers.append(IA_block)

        return nn.Sequential(*reslayers)'''

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x1, x2 = None, None
        if "avg" in self.pool:
            x1 = self.model.avgpool2(x)
        if "max" in self.pool:
            x2 = self.model.maxpool2(x)

        x = x1 if x2 is None else x2 if x1 is None else torch.cat((x1, x2), dim=1)

        f = x.view(x.size(0), -1)
        f = self.bn(f)
        y = self.classifier(f)
        attrs = None
        if self.num_attrs > 0:
            attrs = self.attr_classifier(f)
        return y, attrs


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


if __name__ == "__main__":
    b = torch.randn((5, 3, 224, 224)).cuda()
    model = resnet50(pretrained=True, num_classes=751).cuda()
    out, feat = model(b)
    print(out.shape)
