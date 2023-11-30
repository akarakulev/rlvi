import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils import accuracy


__all__ = ['train_regular']


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'


def train_regular(train_loader, model, optimizer):
    train_total = 0
    train_correct = 0

    for (images, labels, indexes) in train_loader:

        images = Variable(images).to(DEVICE)
        labels = Variable(labels).to(DEVICE)
        
        logits = model(images)
        
        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec

        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = float(train_correct) / float(train_total)
    return train_acc