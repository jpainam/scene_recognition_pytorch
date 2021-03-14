import os
import pickle
import torchvision


class MITIndoor67Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, root: str, transform, with_attribute=False, xi=.8):
        super().__init__(root, transform)
        self.with_attribute = with_attribute
        self.train_val = 'val' if 'val' in root else 'train'

        self.xi = xi
        if with_attribute:
            data = pickle.load(open(os.path.join('/home/paul/datasets/MITIndoor67', f'{self.train_val}_annotations.pkl'), 'rb'))
            self.attribute_images = list(data['images'])
            self.attributes = data['attributes']
            # categories = data['categories']
            self.attribute_labels = data['labels']
            assert len(self.attribute_images) == len(self.imgs)

    def __getitem__(self, idx):
        images, labels = super(MITIndoor67Dataset, self).__getitem__(idx)
        attrs = []
        img_path, class_index = self.imgs[idx]
        if self.with_attribute:
            attrs = self.attribute_labels[self.attribute_images.index(os.path.basename(img_path))]
            #attrs[attrs > self.xi] = 1.
            #attrs[attrs <= self.xi] = 0.

        return images, labels, attrs
