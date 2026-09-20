# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np


# from semilearn.core.hooks import Hook
# from semilearn.algorithms.utils import concat_all_gather


class DistAlignEMAHook():
    """
    Distribution Alignment Hook for conducting distribution alignment
    """

    def __init__(self, num_classes, momentum = 0.999, align_ratio = 0.3, p_target_type = 'uniform', p_target = None):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum
        self.align_ratio = torch.tensor(align_ratio)
        # p_target
        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target)
        print('distribution alignment p_target:', self.p_target)
        # p_model
        self.p_model = None

    @torch.no_grad()
    def dist_align(self, algorithm, epoch, probs_x_ulb = None, probs_x_lb = None):
        batch_size, _, height, width = probs_x_ulb.shape
        # update queue
        self.update_p(probs_x_ulb, probs_x_lb, epoch)
        if self.p_model is not None and not self.p_model.is_cuda:
            self.p_model = self.p_model.to(probs_x_ulb.device)
        # 将 p_target 转换为 [1, num_classes, 1, 1]，然后广播到 [batch_size, num_classes, height, width]
        p_target_expanded = self.p_target.view(1, -1, 1, 1).expand(probs_x_ulb.size(0), self.num_classes,
                                                                   probs_x_ulb.size(2), probs_x_ulb.size(3))
        # 将 p_model 转换为 [1, num_classes, 1, 1]，然后广播到 [batch_size, num_classes, height, width]
        p_model_expanded = self.p_model.view(1, -1, 1, 1).expand(probs_x_ulb.size(0), self.num_classes,
                                                                 probs_x_ulb.size(2), probs_x_ulb.size(3))
        # dist align
        probs_x_ulb = torch.nan_to_num(probs_x_ulb, nan = 0.0)
        no_change_probs = 1 - probs_x_ulb
        probs_x_ulb_aligned = torch.cat((no_change_probs, probs_x_ulb), dim = 1)
        probs_x_ulb_aligned = probs_x_ulb_aligned * (p_target_expanded + 1e-6) / (p_model_expanded + 1e-6)
        # 对每个像素的概率进行归一化处理
        probs_x_ulb_aligned = probs_x_ulb_aligned / (probs_x_ulb_aligned.sum(dim = 1, keepdim = True) + 1e-6)
        # 仅保留变化类别（第二个通道），并将其形状转换为 [batch_size, 1, height, width]
        probs_x_ulb_aligned = probs_x_ulb_aligned[:, 1:2, :, :]
        probs_x_ulb_aligned = torch.nan_to_num(probs_x_ulb_aligned, nan = 0.0)
        return probs_x_ulb_aligned

    @torch.no_grad()
    def update_p(self, probs_x_ulb, probs_x_lb, epoch):
        # 检查设备
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)

        # if algorithm.distributed and algorithm.world_size > 1:
        #     if probs_x_lb is not None and self.update_p_target:
        #         probs_x_lb = concat_all_gather(probs_x_lb)
        #     probs_x_ulb = concat_all_gather(probs_x_ulb)

        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model == None:
            mean_probs_x_ulb = torch.mean(probs_x_ulb, dim = (0, 2, 3))  # [1] 每个像素的平均标签概率
            mean_probs_x_ulb = torch.nan_to_num(mean_probs_x_ulb, nan = 0.0)
            mean_probs_x_ulb = torch.stack([1 - mean_probs_x_ulb, mean_probs_x_ulb], dim = 0)  # [不变，变]
            self.p_model = mean_probs_x_ulb
            # self.p_model = torch.mean(probs_x_ulb, dim = 0)
        else:
            mean_probs_x_ulb = torch.mean(probs_x_ulb, dim = (0, 2, 3))  # [1] 每个像素的平均标签概率
            mean_probs_x_ulb = torch.nan_to_num(mean_probs_x_ulb, nan = 0.0)
            mean_probs_x_ulb = torch.stack([1 - mean_probs_x_ulb, mean_probs_x_ulb], dim = 0)
            # self.p_model = self.p_model * self.m + mean_probs_x_ulb * (1 - self.m)

            # 计算当前 p_model 与新的预测之间的差异度量（例如，KL 散度）
            diff_model = torch.abs(self.p_model - mean_probs_x_ulb).sum()  # 计算 p_model 和 mean_probs_x_ulb 的差异

            # 更新的幅度控制（通过差异来控制更新）
            update_ratio = torch.sigmoid(diff_model)  # 使用 sigmoid 使得差异越大，更新幅度越大，最大为 1
            update_ratio = (1 - self.m) * update_ratio
            self.p_model = self.p_model * self.m + mean_probs_x_ulb * update_ratio

        if self.update_p_target:
            if probs_x_lb is not None:
                mean_probs_x_lb = torch.mean(probs_x_lb, dim = (0, 2, 3))  # [1] 每个像素的平均标签概率
                mean_probs_x_lb = torch.nan_to_num(mean_probs_x_lb, nan = 0.0)
                mean_probs_x_lb = torch.stack([1 - mean_probs_x_lb, mean_probs_x_lb], dim = 0)
                # self.p_target = self.p_target * self.m + mean_probs_x_lb * (1 - self.m)

                # 计算当前 p_target 与新的预测之间的差异度量（例如，KL 散度）
                diff_target = torch.abs(self.p_target - mean_probs_x_lb).sum()  # 计算 p_target 和 mean_probs_x_lb 的差异

                # 更新的幅度控制（通过差异来控制更新）
                update_ratio = torch.sigmoid(diff_target)  # 使用 sigmoid 使得差异越大，更新幅度越大，最大为 1
                update_ratio = (1 - self.m) * update_ratio
                self.p_target = self.p_target * self.m + mean_probs_x_lb * update_ratio
            else:
                # 如果 probs_x_lb 为空，则跳过更新
                # print("Warning: probs_x_lb is None, skipping p_target update")
                pass
        self.adjust_m(epoch)

    def adjust_m(self, epoch):
        # 假设训练总共 100 个 epoch，逐渐增加 m 的值
        # epoch 越大，m 越大，使得更新越平滑
        total_epochs = 10  # 训练的总 epoch 数，可以根据实际情况调整
        self.m = min(0.999, 0.9 + 0.1 * (epoch / total_epochs))  # 让 m 在 [0.9, 0.999] 区间内平滑过渡

    def set_p_target(self, p_target_type = 'uniform', p_target = None):
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        # todo 测试align_ratio值，0.9、0.8、0.7
        update_p_target = False
        if p_target_type == 'uniform':
            # p_target = torch.ones((self.num_classes,)) / self.num_classes
            p_target = torch.stack([1 - self.align_ratio, self.align_ratio], dim = 0).unsqueeze(1)
            # p_target = self.align_ratio
        elif p_target_type == 'model':
            # p_target = torch.ones((self.num_classes,)) / self.num_classes
            p_target = torch.stack([1 - self.align_ratio, self.align_ratio], dim = 0).unsqueeze(1)
            # p_target = self.align_ratio
            update_p_target = True
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)
        return update_p_target, p_target
