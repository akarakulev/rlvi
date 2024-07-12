import os
import argparse
import datetime
import time

import numpy as np

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets import Food101
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import methods
import utils
from models.resnet50 import resnet50


parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, help = 'dir to save result txt files', default='results/')
parser.add_argument('--root_dir', type=str, help = 'dir that stores dataset', default='data/')
parser.add_argument('--lr_init', type=float, default=0.001)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, help='batch_size', default=32)
parser.add_argument('--wd', type=float, help='l2 regularization', default=1e-4)
parser.add_argument('--method', type=str, help='[regular, rlvi, coteaching, jocor, cdr, usdnl, bare]', default='regular')
parser.add_argument('--noise_rate', type=float, help='corruption level, should be less than 1', default=0.1)
parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
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
transform_train = transforms.Compose([
	transforms.Resize(224), 
	transforms.CenterCrop(224), 
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	transforms.RandomRotation(45),
	transforms.RandomAffine(45),
	transforms.ColorJitter(),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

transform_test = transforms.Compose([
	transforms.Resize(256), 
	transforms.CenterCrop(224), 
	transforms.ToTensor(), 
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])


class Food101Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True):
        if train:
            split = 'train'
            transform = transform_train
        else:
            split = 'test'
            transform = transform_test

        self.dataset = Food101(root=root,
                                download=True,
                                split=split,
                                transform=transform)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

train_dataset = Food101Dataset(root=args.root_dir, train=True)
test_dataset = Food101Dataset(root=args.root_dir, train=False)


# For alternative methods:
# create rate_schedule to gradually consider less and less samples
if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate
rate_schedule = np.ones(args.n_epoch) * forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)


# Prepare a structured output
save_dir = f"{args.result_dir}/food101/{args.method}"
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str = f"food101_{args.method}"
txtfile = f"{save_dir}/{model_str}-s{args.seed}.txt"
if os.path.exists(txtfile):
    curr_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    new_dest = f"{txtfile}.bak-{curr_time}"
    os.system(f"mv {txtfile} {new_dest}")


def run():
    # Data Loaders
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    split = int(np.floor(args.split_percentage * len(train_dataset)))
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, 
                                               sampler=train_sampler,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=args.batch_size,
                                             sampler=val_sampler,
                                             num_workers=args.num_workers,
                                             drop_last=False,
                                             pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              pin_memory=True)
    
    # Prepare models and optimizers
    if args.method == 'usdnl':
        model = resnet50(input_channel=3, num_classes=101, pretrained=True, dropout_rate=0.25)
    else:
        model = resnet50(input_channel=3, num_classes=101, pretrained=True)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(
        params=[
            {"params": model.fc_params(), "lr": args.lr_init},
            {"params": model.backbone_params()}
        ], 
        lr=1e-4, weight_decay=args.wd
    )

    if args.method == 'jocor' or args.method == 'coteaching':
        model_sec = resnet50(input_channel=3, num_classes=101, pretrained=True)
        model_sec.to(DEVICE)
        if args.method == 'jocor':
            optimizer = torch.optim.Adam(
                params=[
                    {"params": list(model.fc_params()) + list(model_sec.fc_params()), "lr": args.lr_init},
                    {"params": list(model.backbone_params()) + list(model_sec.backbone_params())}
                ], 
                lr=1e-4, weight_decay=args.wd
            )
        else:
            optimizer_sec = torch.optim.Adam(
                params=[
                    {"params": model_sec.fc_params(), "lr": args.lr_init},
                    {"params": model_sec.backbone_params()}
                ], 
                lr=1e-4, weight_decay=args.wd
        )

    # Schedule for the learning rate
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.2)
    if args.method == 'coteaching':
        scheduler_sec = MultiStepLR(optimizer_sec, milestones=[10, 20, 30], gamma=0.2)
    
    if args.method == 'rlvi':
        sample_weights = torch.ones(len(train_dataset)).to(DEVICE)
        residuals = torch.zeros_like(sample_weights).to(DEVICE)
        overfit = False
        threshold = 0


    # Evaluate init model
    test_acc = utils.evaluate(test_loader, model)
    utils.output_table(epoch=0, n_epoch=args.n_epoch, test_acc=test_acc)
    with open(txtfile, "a") as myfile:
        myfile.write("epoch:\ttime_ep\ttau\tfix\ttrain_acc\tval_acc\ttest_acc\n")
        myfile.write(f"0:\t0\t0\t{False}\t0\t0\t{test_acc:8.4f}\n")

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
            val_acc = utils.evaluate(val_loader, model)
            # Using no regularization by default (as val. score is increasing)
            overfit = False

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
            train_acc = methods.train_bare(train_loader, model, optimizer, num_classes=101)
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

        # Save logs to the file
        with open(txtfile, "a") as myfile:
            myfile.write(f"{int(epoch)}:\t{time_ep:.2f}\t{threshold:.2f}\t{overfit}\t"
                         + f"{train_acc:8.4f}\t{val_acc:8.4f}\t{test_acc:8.4f}\n")


if __name__ == '__main__':
    run()
