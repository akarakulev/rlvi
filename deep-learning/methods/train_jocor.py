# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import accuracy


__all__ = ['train_jocor']


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'


def kl_loss_compute(pred, soft_targets, reduce='none'):
    kl = F.kl_div(
        F.log_softmax(pred, dim=1),
        F.softmax(soft_targets, dim=1), 
        reduction=reduce
    )
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


def loss_jocor(y_1, y_2, t, forget_rate, ind, co_lambda=0.1):
    loss_pick_1 = F.cross_entropy(y_1, t, reduction="none") * (1-co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, t, reduction="none") * (1-co_lambda)
    loss_pick = (
        (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2, reduce='none')
        + co_lambda * kl_loss_compute(y_2, y_1, reduce='none')).cpu()
    )
    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))
    ind_update = ind_sorted[:num_remember]
    # exchange
    loss = torch.mean(loss_pick[ind_update])
    return loss


def train_jocor(train_loader, epoch, model1, model2, optimizer, rate_schedule):
    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0

    for (images, labels, indexes) in train_loader:
        ind = indexes.cpu().numpy().transpose()

        images = Variable(images).to(DEVICE)
        labels = Variable(labels).to(DEVICE)

        # Forward + Backward + Optimize
        logits1 = model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec1

        logits2 = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2 += 1
        train_correct2 += prec2

        loss_1 = loss_jocor(logits1, logits2, labels, rate_schedule[epoch], ind)

        optimizer.zero_grad()
        loss_1.backward()
        optimizer.step()

    train_acc1 = float(train_correct) / float(train_total)
    return train_acc1
