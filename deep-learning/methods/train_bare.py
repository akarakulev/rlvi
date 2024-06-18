import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils import accuracy


__all__ = ['train_bare']


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'


class WeightedCCE(nn.Module):
    """
    Implementing BARE with Cross-Entropy (CCE) Loss
    """

    def __init__(self, k=1, num_class=10, reduction="mean"):
        super(WeightedCCE, self).__init__()

        self.k = k
        self.reduction = reduction
        self.num_class = num_class


    def forward(self, prediction, target_label, one_hot=True):
        EPS = 1e-8
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor),
                               num_classes=self.num_class).to(DEVICE)
        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, EPS, 1-EPS)
        pred_tmp = torch.sum(y_true * y_pred, axis=-1).reshape(-1, 1)

        ## Compute batch statistics
        avg_post = torch.mean(y_pred, dim=0)
        avg_post = avg_post.reshape(-1, 1)
        std_post = torch.std(y_pred, dim=0)
        std_post = std_post.reshape(-1, 1)
        avg_post_ref = torch.matmul(y_true.type(torch.float), avg_post)
        std_post_ref = torch.matmul(y_true.type(torch.float), std_post)
        pred_prun = torch.where((pred_tmp - avg_post_ref >= self.k * std_post_ref),
                                pred_tmp, torch.zeros_like(pred_tmp))

        # prun_idx will tell us which examples are
        # 'trustworthy' for the given batch
        prun_idx = torch.where(pred_prun != 0.)[0]
        if len(prun_idx) != 0:
            prun_targets = torch.argmax(torch.index_select(y_true, 0, prun_idx), dim=1)
            weighted_loss = F.cross_entropy(torch.index_select(prediction, 0, prun_idx), 
                                            prun_targets, reduction=self.reduction)
        else:
            weighted_loss = F.cross_entropy(prediction, target_label)
            
        return weighted_loss


def train_bare(train_loader, model, optimizer, num_classes):
    train_total = 0
    train_correct = 0

    loss_fn = WeightedCCE(k=1, num_class=num_classes, reduction="none")
    for (images, labels, indexes) in train_loader:

        images = Variable(images).to(DEVICE)
        labels = Variable(labels).to(DEVICE)
        
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec

        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    train_acc = float(train_correct) / float(train_total)
    return train_acc
