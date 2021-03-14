from torch import nn
from torch.nn import init
import torch


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
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


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num=1, activ='sigmoid', num_bottleneck=512):
        super(ClassBlock, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        if activ == 'sigmoid':
            classifier += [nn.Sigmoid()]
        elif activ == 'softmax':
            classifier += [nn.Softmax()]
        elif activ == 'none':
            classifier += []
        else:
            raise AssertionError("Unsupported activation: {}".format(activ))
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class Reweigthing(nn.Module):
    def __init__(self, num_attrs):
        super(Reweigthing, self).__init__()
        self.lin1 = nn.Linear(num_attrs, num_attrs, bias=True)
        self.lin2 = nn.Linear(num_attrs, num_attrs, bias=True)
        # self.W2 = nn.Parameter(torch.randn((num_attrs, num_attrs)), requires_grad=True)
        self.lin3 = nn.Linear(num_attrs, num_attrs, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, feat_attrs, attrs):
        feat_attrs, attrs = feat_attrs.float(), attrs.float()
        b, f = attrs.size()
        feat_attrs = feat_attrs.view(b, -1,  f).mean(1)
        # out = self.W2 * attrs.t()
        out = self.relu(self.lin1(feat_attrs)) + self.relu(self.lin2(attrs))
        # out = torch.tanh(out)
        out = self.relu(self.lin3(out))
        return attrs * out

