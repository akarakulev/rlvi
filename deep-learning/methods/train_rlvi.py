import torch
from torch.nn import functional as F
from torch.autograd import Variable
from utils import accuracy


__all__ = ['train_rlvi']


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'


@torch.no_grad()
def update_sample_weights(residuals, weights, tol=1e-3, maxiter=40):
    """
    Optimize Bernoulli probabilities
    
    Parameters
    ----------
        residuals : array,
            shape: len(train_loader) - cross-entropy loss for each training sample.
        weights : array,
            shape: len(train_loader) - Bernoulli probabilities pi: pi_i = probability that sample i is non-corrupted.
            Updated inplace.
    """
    residuals.sub_(residuals.min())
    exp_res = torch.exp(-residuals)
    avg_weight = 0.95
    for _ in range(maxiter):
        ratio = avg_weight / (1 - avg_weight)
        new_weights = torch.div(ratio * exp_res, 1 + ratio * exp_res)
        error = torch.norm(new_weights - weights)
        weights[:] = new_weights
        avg_weight = weights.mean()
        if error < tol:
            break
    weights.div_(weights.max())


def false_negative_criterion(weights, alpha=0.05):
    '''Find threshold from the fixed probability (alpha) of type II error'''
    total_positive = torch.sum(1 - weights)
    beta = total_positive * alpha
    sorted_weights, _ = torch.sort(weights, dim=0, descending=True)
    false_negative = torch.cumsum(1 - sorted_weights, dim=0)
    last_index = torch.sum(false_negative <= beta) - 1
    threshold = sorted_weights[last_index]
    return threshold


def train_rlvi(train_loader, model, optimizer,
               residuals, weights, overfit, threshold):
    """
    Train one epoch: apply SGD updates using Bernoulli probabilities. 
    Thus, optimize variational bound of the marginal likelihood 
    instead of the standard neg. log-likelihood (cross-entropy for classification).
    
    Parameters
    ----------
        residuals : array,
            shape: len(train_loader) - to store cross-entropy loss on each training sample.
            Shared across epochs.
        weights : array,
            shape: len(train_loader) - Bernoulli probabilities: pi_i = proba (in [0; 1]) that sample i is non-corrupted.
            Shared across epochs.
        overfit : bool - flag indicating whether overfitting has started.
        threshold : float in [0; 1] - previous threshold for truncation: pi_i < threshold => pi_i = 0.

    Returns
    -------
        train_acc : float - top-1 accuracy on training samples.
        threshold : float in [0; 1] - updated threshold, based on type II error criterion.
    """

    train_total = 0
    train_correct = 0

    for (images, labels, indexes) in train_loader:
      
        images = Variable(images).to(DEVICE)
        labels = Variable(labels).to(DEVICE)
        
        logits = model(images)
        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec

        loss = F.cross_entropy(logits, labels, reduction='none')
        residuals[indexes] = loss  # save the losses on the current batch
        
        batch_weights = weights[indexes]
        loss = loss * batch_weights  # modify loss with Bernoulli probabilities
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    update_sample_weights(residuals, weights)
    if overfit:
        # Regularization: truncate samples with high probability of corruption
        threshold = max(threshold, false_negative_criterion(weights))
        weights[weights < threshold] = 0

    train_acc = float(train_correct) / float(train_total)
    return train_acc, threshold
