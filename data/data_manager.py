import torch
import torchvision
import torchvision.transforms as transforms
from data.random_erasing import RandomErasing
from PIL import Image
import os.path as osp
from torchvision.transforms.transforms import ToTensor, Lambda, Normalize
from .datasets.ADE20KDataset import ADE20KDataset


def get_data(dataset="MITIndoor67", root=None, train_folder='train', val_folder='val', batch_size=64, ten_crops=False):
    assert root is not None
    train_transform = transforms.Compose([
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0]),
    ])

    if ten_crops:
        val_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.TenCrop((224, 224)),
            Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),  # returns a 4D tensor
            Lambda(lambda crops: torch.stack(
                [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if dataset == "ADE20K":
        train_set = ADE20KDataset(root, folder="training", transform=train_transform)
        val_set = ADE20KDataset(root, folder="validation", transform=val_transform)
    elif dataset == "MITIndoor67":
        train_set = torchvision.datasets.ImageFolder(osp.join(root, train_folder), train_transform)
        val_set = torchvision.datasets.ImageFolder(osp.join(root, val_folder), val_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                             num_workers=4)

    assert len(train_set.classes) == len(val_set.classes)

    print('Dataset loaded!')
    print(f'Train set. Size {len(train_set.imgs)}')
    print(f'Validation set. Size {len(val_set.imgs)}')
    print('Train set number of scenes: {}'.format(len(train_set.classes)))
    print('Validation set number of scenes: {}'.format(len(val_set.classes)))
    return train_loader, val_loader, train_set.classes