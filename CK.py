from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


class CK(data.Dataset):

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
        img, target = self.train_data[index], self.train_labels[index]

        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
         return len(self.train_data)


