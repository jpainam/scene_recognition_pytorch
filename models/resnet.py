import torch
from torch import nn
import torchvision
from torch.nn import init
from torchvision import  models

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'densenet161']

from models.block_model import ClassBlock, Reweighting


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
    }

    def __init__(self, depth, pretrained=True,
                 norm=False, with_attribute=False,
                 with_reweighting=False,
                 backbone='resnet',
                 num_classes=0, stride=1, num_attrs=0):
        super(ResNet, self).__init__()
        assert num_classes != 0, 'The number of classes must be non null'
        self.depth = depth
        self.pretrained = pretrained

        self.norm = norm
        self.num_attrs = num_attrs
        self.num_classes = num_classes
        self.with_attribute = with_attribute
        self.with_reweighting = with_reweighting
        self.backbone = backbone
        if self.with_reweighting:
            self.reweighting = Reweighting(num_attrs=num_attrs)

        if 'resnet' in self.backbone and depth in ResNet.__factory:
            model_ft = ResNet.__factory[depth](pretrained=pretrained)
            self.num_features = 2048
            if stride == 1:
                model_ft.layer4[0].downsample[0].stride = (1, 1)
                model_ft.layer4[0].conv2.stride = (1, 1)
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft
        elif 'densenet' in self.backbone:
            model_ft = models.densenet161(pretrained=pretrained)
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft.features
            self.num_features = 1024
        else:
            raise KeyError(f"Unsupported resnet depth {depth} module, must be [18, 34, 50, 101]")

        self.classifier = ClassBlock(input_dim=self.num_features + (self.num_attrs if self.with_reweighting else 0),
                                     class_num=num_classes, activ='none')
        if self.with_attribute:
            for a in range(self.num_attrs):
                self.__setattr__("attr_%d" % a, ClassBlock(input_dim=self.num_features,
                                                           class_num=1, activ='none'))

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

    def forward(self, x, attrs=None):
        # [B x C x W x H]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # [B x 2048]
        pred_attrs = []
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


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def densenet161(**kwargs):
    return ResNet(161, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


if __name__ == "__main__":
    b = torch.randn((5, 3, 224, 224)).cuda()
    model = resnet50(pretrained=True, num_classes=751).cuda()
    out, feat = model(b)
    print(out.shape)
