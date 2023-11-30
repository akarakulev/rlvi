import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils import accuracy


__all__ = ['train_usdnl']


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'


def loss_fn(logits, labels, forget_rate):
    loss_pick = F.cross_entropy(logits, labels.long(), reduction='none')
    loss_pick = loss_pick.cpu()
    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))
    ind_update = ind_sorted[:num_remember]
    # Exchange
    loss = torch.mean(loss_pick[ind_update])
    return loss


def train_usdnl(train_loader, epoch, model, optimizer, rate_schedule):
    model.train()
    train_total = 0
    train_correct = 0

    for (data, labels, indexes) in train_loader:

        data = Variable(data).to(DEVICE)
        labels = Variable(labels).to(DEVICE)

        # Forward + Backward + Optimize
        logits = model(data)
        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec

        loss = loss_fn(logits, labels, rate_schedule[epoch])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
    train_acc = float(train_correct) / float(train_total)
    return train_acc
