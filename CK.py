from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


class CK(data.Dataset):
    """`CK+ Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

        there are 135,177,75,207,84,249,54 images in data
        we choose 123,159,66,186,75,225,48 images for training
        we choose 12,8,9,21,9,24,6 images for testing
        the split are in order according to the fold number
    """

    def __init__(self, path='./data/train_data.h5', fold = 1, transform=None):
        self.transform = transform
        self.fold = fold # the k-fold cross validation
        self.data = h5py.File(path, 'r', driver='core')

        number = len(self.data['data_label']) #41191
        
        self.train_data = []
        self.train_labels = []
        for i in xrange(number):
            self.train_data.append(self.data['data_pixel'][i])
            self.train_labels.append(self.data['data_label'][i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
         return len(self.train_data)


