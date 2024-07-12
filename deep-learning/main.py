import os
import argparse
import datetime
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiplicativeLR, CosineAnnealingLR

import methods
import data_load
import data_tools
import utils

from models.lenet import LeNet
from models.resnet import ResNet18, ResNet34
# models with dropout for USDNL algorithm:
from models.lenet import LeNetDO
from models.resnet import ResNet18DO, ResNet34DO


parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, help = 'dir to save result txt files', default='results/')
parser.add_argument('--root_dir', type=str, help = 'dir that stores datasets', default='data/')
parser.add_argument('--dataset', type=str, help='[mnist, cifar10, cifar100]', default='mnist')
parser.add_argument('--method', type=str, help='[regular, rlvi, coteaching, jocor, cdr, usdnl, bare]', default='rlvi')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.45)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric, instance]', default='pairflip')
parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4, help='number of subprocesses for data loading')
parser.add_argument('--seed', type=int, default=1)
# For alternative methods
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if DEVICE == "cuda":
    torch.cuda.manual_seed(args.seed)


# Load datasets for training, validation, and testing
if args.dataset == 'mnist':
    input_channel = 1
    num_classes = 10
    args.n_epoch = 100
    args.batch_size = 32
    args.wd = 1e-3
    args.lr_init = 0.01

    if args.method != 'usdnl':
        Model = LeNet
    else:
        Model = LeNetDO  # with dropout

    train_dataset = data_load.Mnist(root=args.root_dir,
                                    download=True,
                                    train=True,
                                    transform=Model.transform_train,
                                    target_transform=data_tools.transform_target,
                                    dataset=args.dataset,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                    split_per=args.split_percentage,
                                    random_seed=args.seed)

    val_dataset = data_load.Mnist(root=args.root_dir,
                                    train=False,
                                    transform=Model.transform_test,
                                    target_transform=data_tools.transform_target,
                                    dataset=args.dataset,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                    split_per=args.split_percentage,
                                    random_seed=args.seed)


    test_dataset = data_load.MnistTest(root=args.root_dir,
                                        transform=Model.transform_test,
                                        target_transform=data_tools.transform_target)


if args.dataset == 'cifar10':
    input_channel = 3
    num_classes = 10
    args.n_epoch = 200
    args.batch_size = 128
    args.wd = 5e-4
    args.lr_init = 0.01

    if args.method != 'usdnl':
        Model = ResNet18
    else:
        Model = ResNet18DO  # with dropout

    train_dataset = data_load.Cifar10(root=args.root_dir,
                                        download=True, 
                                        train=True,
                                        transform=Model.transform_train,
                                        target_transform=data_tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

    val_dataset = data_load.Cifar10(root=args.root_dir,
                                    train=False,
                                    transform=Model.transform_test,
                                    target_transform=data_tools.transform_target,
                                    dataset=args.dataset,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                    split_per=args.split_percentage,
                                    random_seed=args.seed)


    test_dataset = data_load.Cifar10Test(root=args.root_dir,
                                            transform=Model.transform_test,
                                            target_transform=data_tools.transform_target)


if args.dataset == 'cifar100':
    input_channel = 3
    num_classes = 100
    args.n_epoch = 200
    args.batch_size = 128
    args.wd = 5e-4
    args.lr_init = 0.01

    if args.method != 'usdnl':
        Model = ResNet34
    else:
        Model = ResNet34DO  # with dropout

    train_dataset = data_load.Cifar100(root=args.root_dir,
                                        download=True,
                                        train=True,
                                        transform=Model.transform_train,
                                        target_transform=data_tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

    val_dataset = data_load.Cifar100(root=args.root_dir,
                                        train=False,
                                        transform=Model.transform_test,
                                        target_transform=data_tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

    test_dataset = data_load.Cifar100Test(root=args.root_dir, 
                                            transform=Model.transform_test, 
                                            target_transform=data_tools.transform_target)


# For alternative methods:
# create rate_schedule to gradually consider less and less samples
if args.forget_rate is None:
    forget_rate = args.noise_rate
    if args.noise_type == 'asymmetric':
        forget_rate /= 2.
else:
    forget_rate = args.forget_rate
rate_schedule = np.ones(args.n_epoch) * forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)


# Prepare a structured output
save_dir = f"{args.result_dir}/{args.dataset}/{args.method}"
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str = f"{args.dataset}_{args.method}_{args.noise_type}_{args.noise_rate}"
txtfile = f"{save_dir}/{model_str}-s{args.seed}.txt"
if os.path.exists(txtfile):
    curr_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    new_dest = f"{txtfile}.bak-{curr_time}"
    os.system(f"mv {txtfile} {new_dest}")


def run():
    # Data Loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True,
                                               pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            drop_last=False,
                                            shuffle=False,
                                            pin_memory=True)

    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False,
                                              pin_memory=True)
    
    # Prepare models and optimizers
    model = Model(input_channel=input_channel, num_classes=num_classes)
    model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, weight_decay=args.wd, momentum=args.momentum)

    if args.method == 'jocor':
        model_sec = Model(input_channel=input_channel, num_classes=num_classes)
        model_sec.to(DEVICE)
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(model_sec.parameters()), 
            lr=args.lr_init, weight_decay=args.wd, momentum=args.momentum
        )

    # Use Multipliactive LR decay for MNIST and Cosine Annealing for CIFAR10/100
    if args.dataset == 'mnist':
        scheduler = MultiplicativeLR(optimizer, utils.get_lr_factor)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=200)

    if args.method == 'coteaching':
        model_sec = Model(input_channel=input_channel, num_classes=num_classes)
        model_sec.to(DEVICE)
        optimizer_sec = torch.optim.SGD(model_sec.parameters(), lr=args.lr_init, weight_decay=args.wd, momentum=args.momentum)
        if args.dataset == 'mnist':
            scheduler_sec = MultiplicativeLR(optimizer_sec, utils.get_lr_factor)
        else:
            scheduler_sec = CosineAnnealingLR(optimizer_sec, T_max=200)
    

    if args.method == 'rlvi':
        sample_weights = torch.ones(len(train_dataset)).to(DEVICE)
        residuals = torch.zeros_like(sample_weights).to(DEVICE)
        overfit = False
        threshold = 0
        val_acc_old, val_acc_old_old = 0, 0


    # Evaluate init model
    test_acc = utils.evaluate(test_loader, model)
    utils.output_table(epoch=0, n_epoch=args.n_epoch, test_acc=test_acc)
    with open(txtfile, "a") as myfile:
        myfile.write("epoch:\ttime_ep\ttau\tfix\tclean,%\tcorr,%\ttrain_acc\tval_acc\ttest_acc\n")
        myfile.write(f"0:\t0\t0\t{False}\t100\t0\t0\t0\t{test_acc:8.4f}\n")

    # Training
    for epoch in range(1, args.n_epoch):
        model.train()

        time_ep = time.time()

        #### Start one epoch of training with selected method ####

        if args.method == "regular":
            train_acc = methods.train_regular(train_loader, model, optimizer)
            val_acc = utils.evaluate(val_loader, model)

        elif args.method == "rlvi":
            train_acc, threshold = methods.train_rlvi(
                train_loader, model, optimizer,
                residuals, sample_weights, overfit, threshold
            )
            # Check if overfitting has started
            val_acc = utils.evaluate(val_loader, model)
            if not overfit:
                if epoch > 2:
                    # Overfitting started <=> validation score is dropping
                    overfit = (val_acc < 0.5 * (val_acc_old + val_acc_old_old))
                val_acc_old_old = val_acc_old
                val_acc_old = val_acc

        elif args.method == 'coteaching':
            model_sec.train()
            train_acc = methods.train_coteaching(
                train_loader, epoch, 
                model, optimizer, model_sec, optimizer_sec,
                rate_schedule
            )
            val_acc = utils.evaluate(val_loader, model)
            scheduler_sec.step()

        elif args.method == 'jocor':
            model_sec.train()
            train_acc = methods.train_jocor(
                train_loader, epoch, 
                model, model_sec, optimizer, 
                rate_schedule
            )
            val_acc = utils.evaluate(val_loader, model)

        elif args.method == 'cdr':
            train_acc = methods.train_cdr(train_loader, epoch, model, optimizer, rate_schedule)
            val_acc = utils.evaluate(val_loader, model)

        elif args.method == 'usdnl':
            train_acc = methods.train_usdnl(train_loader, epoch, model, optimizer, rate_schedule)
            val_acc = utils.evaluate(val_loader, model)

        elif args.method == 'bare':
            train_acc = methods.train_bare(train_loader, model, optimizer, num_classes)
            val_acc = utils.evaluate(val_loader, model)

        # Update LR
        scheduler.step()

        #### Finish one epoch of training with selected method ####

        # Log info
        time_ep = time.time() - time_ep
        test_acc = utils.evaluate(test_loader, model)

        # Print log-table
        if (epoch + 1) % args.print_freq == 0:
            utils.output_table(epoch, args.n_epoch, time_ep, train_acc=train_acc, test_acc=test_acc)
        else:
            utils.output_table(epoch, args.n_epoch, time_ep, train_acc=train_acc)

        # Prepare output: put dummy values for alternative methods
        if args.method != 'rlvi':
            overfit = False
            threshold = 0
            clean, corr = 1, 0
        # Check the number of correctly identified corrupted samples for RLVI
        if (args.method == 'rlvi') and (args.noise_rate > 0):
            mask = (sample_weights > threshold).cpu()
            clean, corr = utils.get_ratio_corrupted(mask, train_dataset.noise_mask)

        # Save logs to the file
        with open(txtfile, "a") as myfile:
            myfile.write(f"{int(epoch)}:\t{time_ep:.2f}\t{threshold:.2f}\t{overfit}\t"
                         + f"{clean*100:.2f}\t{corr*100:.2f}\t"
                         + f"{train_acc:8.4f}\t{val_acc:8.4f}\t{test_acc:8.4f}\n")


if __name__ == '__main__':
    run()
