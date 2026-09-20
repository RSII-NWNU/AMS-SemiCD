# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import torch.nn as nn
from torch.nn import functional as F


def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


class CELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, logits, targets, reduction='none'):
        return ce_loss(logits, targets, reduction)


def consistency_loss(logits, targets, name='ce', mask=None, mask2=None):
    """
    计算损失函数。

    Args:
        logits (torch.Tensor): 模型的原始输出。
        targets (torch.Tensor): 真实标签。
        name (str): 损失函数名称，可选 'ce', 'mse', 'l1'。
        mask (torch.Tensor, optional): 第一个掩码，用于加权或忽略部分损失。
        mask2 (torch.Tensor, optional): 第二个掩码。

    Returns:
        torch.Tensor: 计算得到的标量损失值。
    """
    assert name in ['ce', 'mse', 'l1']

    if name == 'ce':
        # 适用于二分类分割任务
        # 期望 logits 和 targets 的形状为 [B, 1, H, W]
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        # .squeeze(1) 将 [B, 1, H, W] 变为 [B, H, W] 以匹配 BCEWithLogitsLoss 的输入要求
        loss = bce_loss(logits, targets.float())
    elif name == 'mse':
        # 期望 logits 和 targets 的形状为 [B, C, H, W]，其中 C 是类别数
        # targets 应该是 one-hot 编码的
        probs = torch.softmax(logits, dim=1)  # 在类别维度应用 softmax
        loss = F.mse_loss(probs, targets.float(), reduction='none')
    else:  # name == 'l1'
        # 期望 logits 和 targets 形状匹配
        loss = F.l1_loss(logits, targets.float(), reduction='none')

    # 合并掩码
    final_mask = None
    if mask is not None:
        final_mask = mask.float()
    if mask2 is not None:
        # 如果 final_mask 已存在，则与 mask2 相乘，否则直接使用 mask2
        if final_mask is not None:
            final_mask = final_mask * mask2.float()
        else:
            final_mask = mask2.float()

    # 应用合并后的掩码并计算最终损失
    if final_mask is not None:
        # 如果损失和掩码维度不完全匹配（例如，损失是多通道的），则进行广播
        # [B, C, H, W] * [B, 1, H, W] -> [B, C, H, W]
        if loss.dim() > final_mask.dim():
            final_mask = final_mask.unsqueeze(1)  # 增加通道维度以进行广播

        loss = loss * final_mask

        # 正确的归一化：总损失 / 掩码总和
        # 添加 epsilon 防止除以零
        return loss.sum() / (final_mask.sum() + 1e-8)
    else:
        # 如果没有掩码，直接求平均
        return loss.mean()


class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """

    def forward(self, logits, targets, name = 'ce', mask = None, mask2 = None):
        return consistency_loss(logits, targets, name, mask, mask2)


class TripletContrastiveLoss(nn.Module):
    def __init__(self, temp = 0.1, weights = (0.4, 0.3, 0.3)):
        super().__init__()
        self.temp = temp
        self.weights = weights  # (跨分支权重, 公共特征权重, 同分支权重)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.upsample2 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)

    def _project(self, feat):
        """特征标准化投影"""
        return F.normalize(self.pool(feat).flatten(1), p = 2, dim = 1)

    def forward(self, f_commons, delta_ress, delta_vms):
        """
        输入:
            f_commons: 各层公共特征列表 [4×[B,C,H,W]]
            delta_ress: ResNeXt互补特征列表 [4×[B,C,H,W]]
            delta_vms: Vimamba互补特征列表 [4×[B,C,H,W]]
        返回:
            综合对比损失
        """
        # 特征投影
        z_commons = [self._project(f) for f in f_commons]
        z_ress = [self._project(d) for d in delta_ress]
        z_vms = [self._project(d) for d in delta_vms]

        # 计算三大对比项
        loss_cross = self._cross_branch_contrast(z_ress, z_vms)  # 跨分支对比
        loss_public = self._public_contrast(z_commons, z_ress, z_vms)  # 公共特征对比
        # loss_intra = self._intra_branch_contrast(z_ress, z_vms)  # 同分支对比

        return self.weights[0] * loss_cross + self.weights[1] * loss_public
        # return (self.weights[0] * loss_cross +
        #         self.weights[1] * loss_public +
        #         self.weights[2] * loss_intra)

    def _cross_branch_contrast(self, z_ress, z_vms):
        """跨分支对比：同层不同分支特征互斥"""
        loss = 0
        for z_res, z_vm in zip(z_ress, z_vms):
            sim_matrix = torch.mm(z_res, z_vm.T) / self.temp  # [B,B]
            labels = torch.arange(z_res.size(0), device = z_res.device)

            # 关键修改：对相似度取负，使交叉熵强制对角线最小化
            loss += F.cross_entropy(-sim_matrix, labels)  # 鼓励对角线相似度低
            loss += F.cross_entropy(-sim_matrix.T, labels)  # 对称方向
        return loss / (2 * len(z_ress))  # 平均所有损失项

    def _public_contrast(self, z_commons, z_ress, z_vms):
        """公共特征对比：所有互补特征与公共特征分离"""
        loss = 0
        for z_com, z_res, z_vm in zip(z_commons, z_ress, z_vms):
            labels = torch.arange(z_com.size(0), device = z_res.device)
            # 公共特征作为锚点
            sim_res = torch.mm(z_com, z_res.T) / self.temp  # [B,B]
            sim_vm = torch.mm(z_com, z_vm.T) / self.temp
            # 使用类索引标签
            loss += F.cross_entropy(-sim_res, labels)
            loss += F.cross_entropy(-sim_vm, labels)
        return loss / (2 * len(z_commons))

    # def _intra_branch_contrast(self, z_ress, z_vms):
    #     """同分支对比：同分支跨层级相似"""
    #
    #     def _branch_loss(z_list):
    #         loss = 0
    #         num_layers = len(z_list)
    #         num_pairs = num_layers * (num_layers - 1) // 2
    #         if num_pairs == 0:
    #             return 0.0  # 避免除以零
    #
    #         for i in range(num_layers):
    #             for j in range(i + 1, num_layers):
    #                 sim = torch.mm(self.upsample2(z_list[i]), z_list[j].T) / self.temp  # [B,B]
    #                 labels = torch.arange(z_list[i].size(0), device = sim.device)
    #                 loss += F.cross_entropy(sim, labels)
    #         return loss / num_pairs  # 归一化到层对数
    #
    #     z_ress_A = z_ress[:4]
    #     z_ress_B = z_ress[4:]
    #     z_vms_A = z_vms[:4]
    #     z_vms_B = z_vms[4:]
    #     loss_res_A = _branch_loss(z_ress_A)
    #     loss_res_B = _branch_loss(z_ress_B)
    #     loss_res = loss_res_A + loss_res_B
    #     loss_vm_A = _branch_loss(z_vms_A)
    #     loss_vm_B = _branch_loss(z_vms_B)
    #     loss_vm = loss_vm_A + loss_vm_B
    #
    #     return (loss_res + loss_vm) / 4  # 分支间平均
