# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


# from semilearn.core.hooks import Hook
# from semilearn.algorithms.utils import smooth_targets


class PseudoLabelingHook():
    """
    Pseudo Labeling Hook
    """

    def __init__(self):
        super().__init__()

    def smooth_targets(self, logits, targets, smoothing = 0.1):
        """
        label smoothing
        """
        with torch.no_grad():
            num_classes = 2
            # targets = torch.stack([1 - targets, targets], dim = 1)
            true_dist = torch.zeros_like(logits)
            # 对除了目标类别的其他类别赋予平滑值
            true_dist.fill_(smoothing / (num_classes - 1))
            # true_dist.scatter_(1, targets, (1.0 - smoothing))
            true_dist[targets == 1] = 1.0 - smoothing  # 对应目标标签为 1 的地方，设置为 1 - smoothing
        return true_dist

    @torch.no_grad()
    def gen_ulb_targets(self,
                        *args,
                        logits = None,
                        use_hard_label = True,
                        T = 1.0,
                        threshold = 0.5,
                        softmax = True,  # whether to compute softmax for logits, input must be logits
                        label_smoothing = 0.1):

        """
        generate pseudo-labels from logits/probs

        Args:
            algorithm: base algorithm
            logits: logits (or probs, need to set softmax to False)
            use_hard_label: flag of using hard labels instead of soft labels
            T: temperature parameters
            softmax: flag of using softmax on logits
            label_smoothing: label_smoothing parameter
        """

        logits = logits.detach()
        if use_hard_label:
            # return hard label directly
            # pseudo_label = torch.sigmoid(logits)
            # pseudo_label = (logits > 0).long()
            probability = torch.sigmoid(logits)

            # todo 可优化加入置信度
            # uncertain_mask = (probability > 0.45) & (probability < 0.55)  # 标记不确定区域
            pseudo_label = (probability > threshold).long()  # 使用threshold作为阈值判断变化
            # pseudo_label = probability

            if label_smoothing:
                pseudo_label = self.smooth_targets(logits, pseudo_label, label_smoothing)
            return pseudo_label

        # return soft label
        if softmax:
            pseudo_label = torch.sigmoid(logits / T)
            # pseudo_label = algorithm.compute_prob(logits / T)
        else:
            # inputs logits converted to probabilities already
            pseudo_label = logits
        return pseudo_label
