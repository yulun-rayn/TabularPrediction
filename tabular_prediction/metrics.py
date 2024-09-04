import numpy as np

import torch

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, balanced_accuracy_score, average_precision_score, mean_squared_error, mean_absolute_error, r2_score

def accuracy_metric(target, pred):
    target = target.cpu().numpy() if torch.is_tensor(target) else target
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    if len(np.unique(target)) > 2:
        return accuracy_score(target, np.argmax(pred, -1))
    else:
        return accuracy_score(target, pred[:, 1] > 0.5)

def cross_entropy_metric(target, pred):
    target = target.cpu().numpy() if torch.is_tensor(target) else target
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    return log_loss(target, pred)

def auc_metric(target, pred, multi_class='ovo'):
    target = target.cpu().numpy() if torch.is_tensor(target) else target
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    if len(np.unique(target)) > 2:
        return roc_auc_score(target, pred, multi_class=multi_class)
    else:
        if len(pred.shape) == 2:
            pred = pred[:, 1]
        return roc_auc_score(target, pred)

def balanced_accuracy_metric(target, pred):
    target = target.cpu().numpy() if torch.is_tensor(target) else target
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    if len(np.unique(target)) > 2:
        return balanced_accuracy_score(target, np.argmax(pred, -1))
    else:
        return balanced_accuracy_score(target, pred[:, 1] > 0.5)

def average_precision_metric(target, pred):
    target = target.cpu().numpy() if torch.is_tensor(target) else target
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    if len(np.unique(target)) > 2:
        return average_precision_score(target, np.argmax(pred, -1))
    else:
        return average_precision_score(target, pred[:, 1] > 0.5)


def rmse_metric(target, pred):
    target = target.cpu().numpy() if torch.is_tensor(target) else target
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    return np.sqrt(mean_squared_error(target, pred))

def mse_metric(target, pred):
    target = target.cpu().numpy() if torch.is_tensor(target) else target
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    return mean_squared_error(target, pred)

def mae_metric(target, pred):
    target = target.cpu().numpy() if torch.is_tensor(target) else target
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    return mean_absolute_error(target, pred)


def r2_metric(target, pred):
    target = target.cpu().numpy() if torch.is_tensor(target) else target
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    return r2_score(target, pred)
