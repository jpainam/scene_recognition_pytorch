from .resnet import *
from torch import nn
import torch
from .resnext import *



def get_model(num_classes=0,
              dropout=0.5,
              num_attrs=0,
              with_attribute=False,
              with_reweighting=False,
              num_features=2048,
              arch=50,
              backbone='resnet'
              ):
    if 'resnet' in backbone and arch == 50:
        model = resnet50(pretrained=True,
                         num_classes=num_classes,
                         num_attrs=num_attrs,
                         backbone=backbone,
                         with_reweighting=with_reweighting,
                         with_attribute=with_attribute)
    elif 'resnet' in backbone and arch == 101:
        print("Loading arch: resnet101")
        model = resnet101(pretrained=True,
                          num_classes=num_classes,
                          num_attrs=num_attrs,
                          backbone=backbone,
                          with_reweighting=with_reweighting,
                          with_attribute=with_attribute)
    elif 'densenet' in backbone and arch == 161:
        model = densenet161(pretrained=True,
                            num_classes=num_classes,
                            num_attrs=num_attrs,
                            backbone=backbone,
                            with_reweighting=with_reweighting,
                            with_attribute=with_attribute)
    elif 'resnext' in backbone:
        model = ResNext(num_classes=num_classes)
    else:
        raise Exception('choose the network architecture properly')
    print()
    print("Model Loaded {}{}:".format(backbone, arch))
    print(f"num_features : {num_features}")
    print(f"Model has embedding : {num_features > 0}")
    print(f"Dropout : {dropout}")
    print(f"With Attribute : {with_attribute}")
    print(f"With ARM : {with_reweighting}")
    print(model)
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze a layer
    for param in model.features.layer4[2].parameters():
        param.requires_grad = True
    for param in model.features.layer4[1].parameters():
        param.requires_grad = True

    #total_params = sum(p.numel() for p in model.parameters())
    #print(f'{total_params:,} total parameters.')
    #total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'{total_trainable_params:,} training parameters.')

    return model
