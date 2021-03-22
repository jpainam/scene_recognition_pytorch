from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import random
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pickle

class ADE20KDataset(Dataset):
    """Class for ADE20K dataset."""

    def __init__(self, root, folder, transform=None, with_attribute=False):
        """
        Initialize the dataset
        :param root_dir: Root directory to the dataset
        :param folder: Dataset set: Training or Validation
        """
        # Extract main path and set (Train or Val)
        self.root = root
        self.folder = f"{folder}".lower()
        self._transform = transform

        # Decode dataset scene categories
        self.classes = list()

        fd = open(os.path.join(root, "Scene_Names.txt"))
        for line in fd:
            self.classes.append(line.split()[0])
        self.classes = sorted(self.classes)
        self.nclasses = len(self.classes)

        # Create list for filenames and scene ground-truth labels
        self.imgs = list()
        self.labels = list()
        self.labels_index = list()

        fd = open(os.path.join(root, ("sceneCategories_" + self.folder + ".txt")))
        for line in fd:
            name, label = line.split()
            self.imgs.append(name)
            self.labels.append(label)
            self.labels_index.append(self.classes.index(label))

        # Control Statements for data loading
        assert len(self.imgs) == len(self.labels) == len(self.labels_index)
        assert self.nclasses == 1055
        self.with_attribute = with_attribute
        if with_attribute:
            data = pickle.load(open(os.path.join(root, 'annotations.pkl', 'rb')))
            self.attribute_images = list(data['images'])
            self.attributes = data['attributes']
            # categories = data['categories']
            self.attribute_labels = data['labels']
            assert len(self.attribute_images) == len(self.imgs)

    def __len__(self):
        """
        Function to get the size of the dataset
        :return: Size of dataset
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Function to get a sample from the dataset. First  RGB images are read in PIL format. Then
        transformations are applied from PIL to Numpy arrays to Tensors.

        For regular usage:
            - Images should be outputed with dimensions (3, W, H)
        In the case that 10-crops are used:
            - Images should be outputed with dimensions (10, 3, W, H)

        :param idx: Index
        :return: images , scene category index (labels)
        """

        # Get RGB image path and load it
        img_name = os.path.join(self.root, "images", self.folder, (self.imgs[idx] + ".jpg"))
        img = Image.open(img_name)

        # Convert it to RGB if gray-scale
        if img.mode is not "RGB":
            img = img.convert("RGB")

        if self._transform is not None:
            img = self._transform(img)

        assert self.labels_index[idx] == self.classes.index(self.labels[idx])
        attrs = []
        if self.with_attribute:
            attrs = self.attribute_labels[self.attribute_images.index(self.labels[idx])]
        return img, self.labels_index[idx], attrs
