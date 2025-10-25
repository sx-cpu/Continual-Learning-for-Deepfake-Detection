import torch
import os
import numpy as np

from dataloaders.wrapper import Subclass, AppendName, CacheClassLabel1, CacheClassLabel_multi
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



def get_total_Data_multi(opt, remap_class=False):
    class_list = range(2*len(opt.task_name))
    # dataset_splits = {}
    for id, name in enumerate(opt.task_name):
        dataset = get_dataset1(opt, name, id)
        dataset = CacheClassLabel_multi(dataset)
        if id == 0:
            dataset_total = dataset
        else:
            dataset_total = torch.utils.data.ConcatDataset((dataset_total, dataset))
    dataset_total = CacheClassLabel1(dataset_total)

    return dataset_total

def get_tos_multi(opt):
    task_output_space = {}
    # for name in opt.task_name:
    task_output_space['All'] = len(opt.task_name)*2
    return task_output_space