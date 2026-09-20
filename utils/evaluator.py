import numpy as np
import torch

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        # 混淆矩阵保持在 CPU 端 (NumPy)，因为它只有 num_class x num_class 大小，不占空间
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def reset(self):
        """清空混淆矩阵，为下一个 Epoch 做准备"""
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_tp_fp_tn_fn(self):
        """统一返回各类的 TP, FP, TN, FN (均为数组格式，无需区分二分类和多分类)"""
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, _, _ = self.get_tp_fp_tn_fn()
        return tp / (tp + fp + 1e-6)

    def Recall(self):
        tp, _, _, fn = self.get_tp_fp_tn_fn()
        return tp / (tp + fn + 1e-6)

    def F1(self):
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        return (2 * precision * recall) / (precision + recall + 1e-6)

    def OA(self):
        """总体精度 (标量)"""
        return np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-6)

    def Kappa(self):
        """Kappa 系数 (标量)"""
        cm = self.confusion_matrix
        total = cm.sum()
        if total == 0:
            return 0.0
        po = np.diag(cm).sum() / total
        pe = (cm.sum(axis=1) @ cm.sum(axis=0)) / (total ** 2)
        return 0.0 if pe >= 1.0 else (po - pe) / (1 - pe + 1e-6)

    def False_Alarm_Rate(self):
        _, fp, tn, _ = self.get_tp_fp_tn_fn()
        return fp / (fp + tn + 1e-6)

    def Missed_Detection_Rate(self):
        tp, _, _, fn = self.get_tp_fp_tn_fn()
        return fn / (tp + fn + 1e-6)

    def Intersection_over_Union(self):
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        return tp / (tp + fp + fn + 1e-6)

    def Mean_Intersection_over_Union(self):
        return np.nanmean(self.Intersection_over_Union())

    # ===================================================================
    # 核心性能优化区
    # ===================================================================
    def _generate_matrix_numpy(self, gt_image, pre_image):
        """备用的 NumPy 版计算（如果传入的是 numpy 数组）"""
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        pre_image = np.clip(pre_image, 0, self.num_class - 1)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        return count.reshape(self.num_class, self.num_class)

    def _generate_matrix_torch(self, gt_image, pre_image):
        """🚀 GPU 原生计算：直接在显存中统计，彻底释放 CPU 瓶颈"""
        
        # === 新增：强制将输入转换为 64位整型 (LongTensor) ===
        gt_image = gt_image.long()
        pre_image = pre_image.long()
        # ===============================================
        
        # 限制预测值范围，防止越界
        pre_image = torch.clamp(pre_image, 0, self.num_class - 1)
        
        gt_flat = gt_image.flatten()
        pre_flat = pre_image.flatten()
        
        mask = (gt_flat >= 0) & (gt_flat < self.num_class)
        gt_valid = gt_flat[mask]
        pre_valid = pre_flat[mask]
        
        label = self.num_class * gt_valid + pre_valid
        
        # 现在 label 是整型了，bincount 可以在 GPU 上愉快地奔跑了
        count = torch.bincount(label, minlength=self.num_class ** 2)
        
        # 计算完之后，只把最终的小矩阵拉回 CPU
        return count.reshape(self.num_class, self.num_class).cpu().numpy()

    def add_batch(self, gt_image, pre_image):
        """自动识别传入的是 Tensor 还是 Numpy Array"""
        assert gt_image.shape == pre_image.shape, f"Shape mismatch: {gt_image.shape} vs {pre_image.shape}"
        
        # 智能分流：是 Tensor 就用 GPU 算，是 Array 就用 CPU 算
        if isinstance(gt_image, torch.Tensor):
            self.confusion_matrix += self._generate_matrix_torch(gt_image, pre_image)
        else:
            self.confusion_matrix += self._generate_matrix_numpy(gt_image, pre_image)