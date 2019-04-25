import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score


def compute_AUCs(gt, pred, n_classes):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_classes):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def weighted_binary_cross_entropy(output, target, pos_weights, neg_weights):
    loss_arr = F.binary_cross_entropy(output, target, reduction="none")
    loss_arr *= (target == 1).float()*pos_weights + \
        (target == 0).float()*neg_weights
    return loss_arr.mean()


def weighted_binary_cross_entropy2(output, target, class_weights):
    loss_arr = F.binary_cross_entropy(output, target, reduction="none")
    loss_arr = loss_arr.mean(0)
    loss_arr *= class_weights
    return loss_arr.mean()
