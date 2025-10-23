import torch
import os
import numpy as np


from .datasets import dataset_folder


def get_dataset1(opt, name, id):
    dset_lst = []
    if opt.isTrain:
        # root = opt.dataroot + '/' + name + '/{}/'.format(opt.train_split)
        root_ = opt.dataroot + '/' + name + '/{}/'.format(opt.train_split)



def get_dataset1(opt, name, id):
    dset_lst = []
    if opt.isTrain:
        # root = opt.dataroot + '/' + name + '/{}/'.format(opt.train_split)
        root_ = opt.dataroot + '/' + name + '/{}/'.format(opt.train_split)
        opt.classes = os.listdir(root_) if opt.multiclass[id] else ['']
        for cls in opt.classes:
            root = root_ + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
    else:
        # root = opt.dataroot + '/' + name + '/{}/'.format(opt.val_split)
        root_ = opt.dataroot + '/' + name + '/{}/'.format(opt.val_split)
        opt.classes = os.listdir(root_) if opt.multiclass[id] else ['']
        for cls in opt.classes:
            root = root_ + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
    # dset = dataset_folder(opt, root)
    # dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)



