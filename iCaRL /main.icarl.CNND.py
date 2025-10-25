import time
from functools import partial
from typing import Callable, Tuple, List
from collections import OrderedDict

import numpy as np
import torch
from math import ceil
from torch import Tensor
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets.cifar import CIFAR100
from cl_dataset_tools import NCProtocol, NCProtocolIterator, TransformationDataset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms

from cl_strategies import icarl_accuracy_measure, icarl_cifar100_augment_data
from cl_strategies.icarl import icarl_accuracy_measure_to_binary, icarl_accuracy_measure_binary
from models import make_icarl_net
from cl_metrics_tools import get_accuracy, get_accuracy_binary
from models.icarl_net import IcarlNet, initialize_icarl_net
from utils import get_dataset_per_pixel_mean, make_theano_training_function, make_theano_validation_function,  \
    make_theano_feature_extraction_function, make_theano_inference_function, make_batch_one_hot

from options.train_options import TrainOptions
from data import *
from CNND_model import Model_CNND
from options.train_options import TrainOptions
from utils.theano_utils import make_theano_training_function_add_binary,make_theano_training_function_binary,make_theano_validation_function_binary,  \
    make_theano_training_function_mixup,make_theano_training_function_ls,make_theano_validation_function_to_binary



def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


def main():
    # This script tries to reproduce results of official iCaRL code
    # https://github.com/srebuffi/iCaRL/blob/master/iCaRL-TheanoLasagne/main_cifar_100_theano.py

    args = TrainOptions().parse()
    ######### Modifiable Settings ##########
    batch_size = args.batch_size # Batch size
    # n          = 5              # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
    # nb_val     = 0            # Validation samples per class
    nb_cl      = 2             # Classes per group
    nb_protos  = args.nb_protos             # Number of prototypes per class at the end: total protoset memory/ total number of classes
    epochs     = int(args.num_epochs)     # Total number of epochs
    lr_old     = args.init_lr        # Initial learning rate 0.0005
    lr_strat   = args.schedule       # Epochs where learning rate gets decreased
    lr_factor  = 5.             # Learning rate decrease factor
    wght_decay = 0.00001        # Weight Decay
    nb_runs    = 1              # Number of runs (random ordering of classes at each run)
    # torch.manual_seed(1993)     # Fix the random seed
    ########################################

    # fixed_class_order = [87,  0, 52, 58, 44, 91, 68, 97, 51, 15,
    #                      94, 92, 10, 72, 49, 78, 61, 14,  8, 86,
    #                      84, 96, 18, 24, 32, 45, 88, 11,  4, 67,
    #                      69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
    #                      17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
    #                       1, 28,  6, 46, 62, 82, 53,  9, 31, 75,
    #                      38, 63, 33, 74, 27, 22, 36,  3, 16, 21,
    #                      60, 19, 70, 90, 89, 43,  5, 42, 65, 76,
    #                      40, 30, 23, 85,  2, 95, 56, 48, 71, 64,
    #                      98, 13, 99,  7, 34, 55, 54, 26, 35, 39]

    # fixed_class_order = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    train_dataset_splits = get_total_Data_multi(args)
    val_opt = get_val_opt()
    val_dataset_splits = get_total_Data_multi(val_opt)
    task_output_space = get_tos_multi(args)
    print(task_output_space)

    task_names = args.task_name
    print('Task order:', task_names)
    task_num = len(task_names)
    class_num = task_num*2

    fixed_class_order = list(range(task_num*2))
  
    protocol = NCProtocol(train_dataset_splits,
                          val_dataset_splits,
                          n_tasks=len(task_names), shuffle=True, seed=None, fixed_class_order=fixed_class_order)
    
    if args.binary:
        model = Model_CNND(1, args)
    else:
        model = Model_CNND(class_num, args)
    model = model.to(device)

    criterion = BCELoss()  # Line 66-67

    # Line 74, 75
    # Note: sh_lr is a theano "shared"
    sh_lr = lr_old

    # noinspection PyTypeChecker
    val_fn: Callable[[Tensor, Tensor],
                     Tuple[Tensor, Tensor, Tensor]] = partial(make_theano_validation_function, model,
                                                              BCELoss(), 'feature_extractor',
                                                              device=device)

    val_fn_to_binary: Callable[[Tensor, Tensor],
                     Tuple[Tensor, Tensor, Tensor]] = partial(make_theano_validation_function_to_binary, model,
                                                              BCELoss(), 'feature_extractor',
                                                              device=device)
    
    val_fn_binary: Callable[[Tensor, Tensor],
                    Tuple[Tensor, Tensor, Tensor]] = partial(make_theano_validation_function_binary, model,
                                                            BCELoss(), 'feature_extractor',
                                                            device=device)

    # noinspection PyTypeChecker
    function_map: Callable[[Tensor], Tensor] = partial(make_theano_feature_extraction_function, model,
                                                       'feature_extractor', device=device, batch_size=batch_size)

    # Lines 90-97: Initialization of the variables for this run
    # dictionary_size = 500 ###????
    

