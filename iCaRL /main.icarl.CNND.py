
import numpy as np
import torch
from options.train_options import TrainOptions


from data import *



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
    