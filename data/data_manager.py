import torch
import torchvision
import torchvision.transforms as transforms
from data.random_erasing import RandomErasing
from PIL import Image
import os.path as osp


def get_data(root=None, train_folder='train', val_folder='val', batch_size=64):
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
    val_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.ImageFolder(osp.join(root, train_folder), train_transform)
    val_set = torchvision.datasets.ImageFolder(osp.join(root, val_folder), val_transform)


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                             num_workers=4)

    assert len(train_set.classes) == len(val_set.classes)

    print('Dataset loaded!')
    print(f'Train set. Size {len(train_set.imgs)}')
    print(f'Validation set. Size {len(val_set.imgs)}')
    print('Train set number of scenes: {}'.format(len(train_set.classes)))
    print('Validation set number of scenes: {}'.format(len(val_set.classes)))
    return train_loader, val_loader, train_set.classes