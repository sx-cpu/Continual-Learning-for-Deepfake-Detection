import imp
from os import path
import torch
import torch.utils.data as data
import numpy as np


class CacheClassLabel1(data.Dataset):
    """
    A dataset wrapper that has a quick access to all labels of data.
    """

    def __init__(self, dataset):
        super(CacheClassLabel1, self).__init__()
        self.dataset = dataset
        self.labels = torch.LongTensor(len(dataset)).fill_(-1)
        for i, data_ in enumerate(dataset):
            self.labels[i] = data_[1]
        self.number_classes = len(torch.unique(self.labels))
        self.targets = self.labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return img, target


class CacheClassLabel_multi(data.Dataset):
    """
    A dataset wrapper thsat has a quick access to all labels of data.
    """

    def __init__(self, dataset, id):
        super(CacheClassLabel_multi, self).__init__()
        self.dataset = dataset
        self.id = id
        self.labels = torch.LongTensor(len(dataset)).fill_(-1)
        self.data_list = []
        for i, data_ in enumerate(dataset):
            self.labels[i] = data_[1] + id*2
            self.data_list.append(data_[0]) 
        self.number_classes = len(torch.unique(self.labels))
        self.targets = self.number_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target_ = self.dataset[index]
        target = self.labels[index]
        return img, target

    def get_image_class(self, label):
        list_label1 = []
        # list_label = [np.array(self.dataset[idx]) for idx, k in enumerate(self.labels) if k==label]
        list_label1 = [self.data_list[idx] for idx, k in enumerate(self.labels) if k==label]
        list_label = torch.stack(list_label1)
        return list_label