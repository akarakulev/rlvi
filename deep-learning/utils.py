import tabulate
import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'


# Multiplicative LR schedule for SGD
def get_lr_factor(epoch):
    """
    Compute multiplicative factor to decay learning rate. 
    To use for MNIST within 100 epochs
    """
    t = epoch / 100
    lr_ratio = 0.01
    if t <= 0.2:
        factor = 1.0
    elif t <= 0.4:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.2) / 0.2
    else:
        factor = lr_ratio
    return factor


# Print log table
def output_table(epoch, n_epoch, time_ep=None, train_acc=None, test_acc=None):
    columns = ['ep', 'time_ep', 'tr_acc', 'te_acc']
    
    values = [f"{epoch}/{n_epoch}", time_ep, train_acc, test_acc]
    table = tabulate.tabulate(
        [values], columns, tablefmt='simple',
        stralign='center', numalign='center', floatfmt=f'8.2f'
    )
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)


# Evaluate a single model
def evaluate(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = Variable(images).to(DEVICE)
            labels = Variable(labels).to(DEVICE)
            logits = model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum()
        acc = 100 * float(correct) / float(total)
    return acc


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Ratio of identified corrupted vs clean for debug
def get_ratio_corrupted(mask, noise_mask):
    clean_pred =(mask == True)
    clean_true = noise_mask
    clean_found = np.logical_and(clean_pred, clean_true)
    clean_found_ratio = np.count_nonzero(clean_found) / np.count_nonzero(clean_true)
    corrupted_pred = (mask == False)
    corrupted_true = (noise_mask == False)
    corrupted_found = np.logical_and(corrupted_pred, corrupted_true)
    corrupted_found_ratio = np.count_nonzero(corrupted_found) / np.count_nonzero(corrupted_true)
    return clean_found_ratio, corrupted_found_ratio