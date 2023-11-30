import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils import accuracy


__all__ = ['train_coteaching']


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'


# Loss function
def loss_coteaching(y_1, y_2, t, forget_rate, ind):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.data.cpu()).to(DEVICE)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.data.cpu()).to(DEVICE)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember


# Train the Co-teaching Model
def train_coteaching(train_loader, epoch, 
                     model1, optimizer1, 
                     model2, optimizer2, 
                     rate_schedule):
    train_total = 0
    train_correct = 0 
    train_total2 = 0
    train_correct2 = 0 

    for (images, labels, indexes) in train_loader:
        ind = indexes.cpu().numpy().transpose()
      
        images = Variable(images).to(DEVICE)
        labels = Variable(labels).to(DEVICE)
        
        logits1 = model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec1

        logits2 = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2 += 1
        train_correct2 += prec2

        loss_1, loss_2 = loss_coteaching(
            logits1, logits2, labels, rate_schedule[epoch], ind
        )

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

    train_acc1 = float(train_correct) / float(train_total)
    return train_acc1