import torch
import torchvision
import torchvision.transforms as transforms
from data.random_erasing import RandomErasing
from PIL import Image
import os.path as osp
from torchvision.transforms.transforms import ToTensor, Lambda, Normalize
from .datasets.ADE20KDataset import ADE20KDataset
from .datasets.MITIndoor67Dataset import MITIndoor67Dataset
from .datasets.SUN397Dataset import SUN397Dataset
from .data_augmentation import PowerPIL
from collections import defaultdict
import numpy as np
import os
import pickle

def get_data(dataset="MITIndoor67", root=None, train_folder='train',
             val_folder='val', batch_size=64, ten_crops=False,
             with_attribute=False):
    assert root is not None
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        PowerPIL(),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        # transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        # transforms.CenterCrop((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0]),
    ])

    if ten_crops:
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.TenCrop((224, 224)),
            Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),  # returns a 4D tensor
            Lambda(lambda crops: torch.stack(
                [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
        ])
    else:
        val_transform = transforms.Compose([
            # transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            # transforms.CenterCrop((224, 224)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if dataset == "ADE20K":
        train_set = ADE20KDataset(root, folder="training", transform=train_transform, with_attribute=with_attribute)
        val_set = ADE20KDataset(root, folder="validation", transform=val_transform, with_attribute=with_attribute)
        assert len(train_set) == 20210
        assert len(val_set) == 2000
        assert len(train_set.classes) == 1055

    elif dataset == "MITIndoor67":
        train_set = MITIndoor67Dataset(osp.join(root, train_folder), train_transform, with_attribute=with_attribute)
        val_set = MITIndoor67Dataset(osp.join(root, val_folder), val_transform, with_attribute=with_attribute)
        assert len(val_set) == 20 * 67
        assert len(train_set) == 80 * 67
        assert len(train_set.classes) == 67
    elif dataset == "SUN397":
        train_set = SUN397Dataset(osp.join(root, train_folder),
                                  train_transform,
                                  with_attribute=with_attribute)

        val_set = SUN397Dataset(osp.join(root, val_folder),
                                val_transform,
                                with_attribute=with_attribute)
        assert (len(train_set) == 50 * 397)
        assert (len(val_set) == 50 * 397)
        assert len(train_set.classes) == 397

    #weigths = get_attr_weight(train_set)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4)

    assert len(train_set.classes) == len(val_set.classes)

    print('Dataset loaded!')
    print(f'Train set. Size {len(train_set.imgs)}')
    print(f'Validation set. Size {len(val_set.imgs)}')
    print('Train set number of scenes: {}'.format(len(train_set.classes)))
    print('Validation set number of scenes: {}'.format(len(val_set.classes)))
    return train_loader, val_loader, train_set.classes, train_set.attributes if with_attribute else []


def get_attr_weight(data):
    '''
    compute pos_weight negative / positive
    '''
    if os.path.exists('./attrs_freq'):
        attrs_freq = pickle.load(open("./attrs_freq", 'rb'))
    else:
        attrs_freq = torch.zeros(134)
        for x, y, att in data:
            tmp = torch.from_numpy(att).clone()
            tmp[tmp > 0.8] = 1.
            tmp[tmp <= 0.8] = 0.
            attrs_freq += tmp
        pickle.dump(attrs_freq.data, open("./attrs_freq", 'wb'))

    # compute the weighs
    N = attrs_freq.sum().item() / len(data)
    all_attrs = torch.empty_like(attrs_freq).fill_(len(data))
    neg_attrs = all_attrs - attrs_freq
    weight = neg_attrs / (attrs_freq + 1)
    print(weight)
    print(min(weight))
    print(max(weight))
    print(min(attrs_freq))
    #print(neg_attrs)
    exit()
    print(neg_attrs/ (attrs_freq + 1e-8))
    exit()
    print("Sum is ")
    print(N)
    wt_per_class = [0.] * 134
    for i in range(67):
        wt_per_class[i] = N / (attrs_freq[i].item() + 1e-8)

    print(wt_per_class)
    exit()
    weight = [0] * len(data)
    #for x, y in zip(train_set, labels):
    #    pass

    return weight
