import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss_functions.ce_loss import RobustCrossEntropyLoss
from utils.loss_functions.dice_loss import SoftDiceLoss
from utils.loss_functions.focal_loss import FocalLoss
from utils.loss_functions.loss_autoweight import AutomaticWeightedLoss


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        y_pred = F.sigmoid(y_pred)
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b

def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)

########## Uncertainty-Aware loss (DC + CE + Focal) ##########
class AutoWeighted_DC_and_CE_and_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, focal_kwargs, ignore_label = None, dice_class = SoftDiceLoss):
        """
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param focal_kwargs:
        """
        super(AutoWeighted_DC_and_CE_and_Focal_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin = softmax_helper_dim1, **soft_dice_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.focal = FocalLoss(apply_nonlin = softmax_helper_dim1, **focal_kwargs)
        self.awl = AutomaticWeightedLoss(3)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[
                       1] == 1, 'ignore label is not implemented for one hot encoded target variables (AutoWeighted_DC_and_CE_and_Focal_loss)'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        target_focal = target[:, 0].long()

        dc_loss = self.dc(net_output, target_dice, loss_mask = mask)
        ce_loss = self.ce(net_output, target[:, 0])
        focal_loss = self.focal(net_output, target_focal)
        result = self.awl(dc_loss, ce_loss, focal_loss)
        return result