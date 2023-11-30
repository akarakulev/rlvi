import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils import accuracy


__all__ = ['train_cdr']


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'


def train_one_step(model, data, label, optimizer, criterion, nonzero_ratio, clip):
    model.train()
    pred = model(data)
    loss = criterion(pred, label)
    loss.backward()
    
    to_concat_g = []
    to_concat_v = []
    for name, param in model.named_parameters():
        if param.dim() in [2, 4]:
            to_concat_g.append(param.grad.data.view(-1))
            to_concat_v.append(param.data.view(-1))
    all_g = torch.cat(to_concat_g)
    all_v = torch.cat(to_concat_v)
    metric = torch.abs(all_g * all_v)
    num_params = all_v.size(0)
    nz = int(nonzero_ratio * num_params)
    top_values, _ = torch.topk(metric, nz)
    thresh = top_values[-1]

    if str(DEVICE) == 'cuda':
       dtype = torch.cuda.FloatTensor
    else:
       dtype = torch.FloatTensor
    for name, param in model.named_parameters():
        if param.dim() in [2, 4]:
            mask = (torch.abs(param.data * param.grad.data) >= thresh).type(dtype)
            mask = mask * clip
            param.grad.data = mask * param.grad.data

    optimizer.step()
    optimizer.zero_grad()
    acc = accuracy(pred, label, topk=(1, 5))

    return float(acc[0]), loss


def train_cdr(train_loader, epoch, model, optimizer, rate_schedule):
    train_total = 0
    train_correct = 0

    clip = 1 - rate_schedule[epoch]
    for (data, labels, indexes) in train_loader:
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)
        # Forward + Backward + Optimize
        logits = model(data)
        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec
        # Loss transfer
        prec, loss = train_one_step(
            model, data, labels, optimizer, nn.CrossEntropyLoss(), clip, clip
        )
      
    train_acc = float(train_correct) / float(train_total)
    return train_acc
