import os
import pickle
import torchvision
from scipy.io import loadmat
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import glob


NUM_CLASSES = 397
NUM_TRAIN_IMAGES = 14340
NUM_TEST_IMAGES = 19850
# sun397/standard-part1-120k
SUN_ATTRIBUTES_CLASSES = 102



class SUN397Dataset(Dataset):
    def __init__(self, root: str, transform, with_attribute=False):
        self.with_attribute = with_attribute
        self.train_val = 'val' if 'val' in root else 'train'
        self._transform = transform
        self.root = root

        if with_attribute:
        #    pass
        #if 'train' in self.train_val:
            '''data = pickle.load(open(os.path.join('/home/paul/datasets/SUN397', f'{self.train_val}_annotations.pkl'), 'rb'))
            self.attribute_images = list(data['images'])
            self.attributes = data['attributes']
            # categories = data['categories']
            self.attribute_labels = data['labels']
            assert len(self.attribute_images) == len(self.imgs)'''

            attrs = loadmat('/home/paul/datasets/SUN397/SUNAttributeDB/attributeLabels_continuous.mat')
            images = loadmat('/home/paul/datasets/SUN397/SUNAttributeDB/images.mat')
            classes = loadmat('/home/paul/datasets/SUN397/SUNAttributeDB/attributes.mat')
            classes = np.array(classes['attributes']).squeeze()
            self.classes = [x[0] for x in classes]
            assert len(self.classes) == SUN_ATTRIBUTES_CLASSES

            images = np.array(images['images']).squeeze()

            self.imgs = np.array([x[0] for x in images])

            attrs = attrs['labels_cv']
            attrs = np.where(attrs > 0, 1, 0)
            self.attributes = attrs.astype(np.float32)

            if self.train_val == 'train':
                assert len(images) == NUM_TRAIN_IMAGES
                assert len(attrs) == len(images)
            assert len(self.imgs) == NUM_TRAIN_IMAGES
        '''if 'val' in self.train_val:
            images = glob.glob(os.path.join(self.root, '*', '*.jpg'))
            self.imgs = [x[len(self.root) + 1:] for x in images]
            assert len(self.imgs) == NUM_TEST_IMAGES
            classes = [x.split("/")[-2] for x in images]
            self.classes = classes'''

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        # Get RGB image path and load it

        img = Image.open(os.path.join(self.root, img))
        assert img is not None

        # Convert it to RGB if gray-scale
        if img.mode is not "RGB":
            img = img.convert("RGB")

        if self._transform is not None:
            img = self._transform(img)
        labels = self.attributes[idx]

        return img, labels


class SUN397DatasetAttribute(Dataset):
    def __init__(self, root: str, transform, with_attribute=False):
        self.train_val = 'val' if 'val' in root else 'train'
        self._transform = transform
        self.root = root

        '''data = pickle.load(open(os.path.join('/home/paul/datasets/SUN397', f'{self.train_val}_annotations.pkl'), 'rb'))
        self.attribute_images = list(data['images'])
        self.attributes = data['attributes']
        # categories = data['categories']
        self.attribute_labels = data['labels']
        assert len(self.attribute_images) == len(self.imgs)'''

        attrs = loadmat('/home/paul/datasets/SUN397/SUNAttributeDB/attributeLabels_continuous.mat')
        images = loadmat('/home/paul/datasets/SUN397/SUNAttributeDB/images.mat')
        classes = loadmat('/home/paul/datasets/SUN397/SUNAttributeDB/attributes.mat')
        classes = np.array(classes['attributes']).squeeze()
        self.classes = [x[0] for x in classes]
        assert len(self.classes) == SUN_ATTRIBUTES_CLASSES

        images = np.array(images['images']).squeeze()

        self.imgs = np.array([x[0] for x in images])

        attrs = attrs['labels_cv']
        self.attributes = attrs

        if self.train_val == 'train':
            assert len(images) == NUM_TRAIN_IMAGES
            assert len(attrs) == len(images)
        assert len(self.imgs) == NUM_TRAIN_IMAGES
    '''if 'val' in self.train_val:
        images = glob.glob(os.path.join(self.root, '*', '*.jpg'))
        self.imgs = [x[len(self.root) + 1:] for x in images]
        assert len(self.imgs) == NUM_TEST_IMAGES
        classes = [x.split("/")[-2] for x in images]
        self.classes = classes'''

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        # Get RGB image path and load it

        img = Image.open(os.path.join(self.root, img))
        assert img is not None

        # Convert it to RGB if gray-scale
        if img.mode is not "RGB":
            img = img.convert("RGB")

        if self._transform is not None:
            img = self._transform(img)
        labels = self.attributes[idx]

        return img, labels
