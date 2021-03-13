from .resnet import *
from torch import nn
import torch


def get_model(num_classes=0,
              dropout=0.5,
              num_attrs=0,
              with_attribute=False,
              num_features=2048
              ):

    model = resnet50(pretrained=True,
                     num_classes=num_classes,
                     num_attrs=num_attrs,
                     with_attribute=with_attribute,
                     num_features=int(num_features))
    print()
    print("Model Loaded:")
    print(f"num_features : {num_features}")
    print(f"Model has embedding : {num_features > 0}")
    print(f"Dropout : {dropout}")
    print(f"With Attribute : {with_attribute}")
    # print(model)
    '''if use_cuda:
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)'''
    return model


def get_optimizer(model):
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = torch.optim.SGD([{"params": base_params, "lr": 0.001},
                                 {"params": model.classifier.parameters(), "lr": 0.01}],
                                momentum=0.9, weight_decay=5e-4, nesterov=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    use_cuda = torch.cuda.is_available()




    epochs = CONFIG['TRAINING']['EPOCH']
    dataloader = {"train": train_loader, "val": val_loader}



def get_criterion():
    criterion = [nn.CrossEntropyLoss(), nn.MultiLabelSoftMarginLoss()]
    return criterion