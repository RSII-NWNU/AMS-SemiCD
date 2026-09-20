import numpy as np
import torch
import math
from torch import Tensor


class AdaptiveStatisticalPseudoLabelRefinement:
    """
    多分类变化检测的动态高斯阈值模块
    输入:
      - logits_or_probs: [B, C, H, W]
      - is_logits=True 表示输入是 logits；否则为概率
    输出:
      - mask: [B, H, W]，值域(0,1]
    """

    def __init__(
            self,
            num_classes: int,
            n_sigma: float = 2.0,
            momentum: float = 0.999,
            per_class: bool = False,
            eps: float = 1e-6,
            min_thr_base: float = 0.3,
            max_thr_base: float = 0.95,
            warmup_epochs: int = 5,
            use_dynamic_threshold: bool = True,
            use_soft_weighting: bool = True,
            enable_monitor: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_sigma = float(n_sigma)
        self.m_base = float(momentum)
        self.per_class = per_class
        self.min_mu = 0.1
        self.eps = float(eps)
        self.min_thr_base = float(min_thr_base)
        self.max_thr_base = float(max_thr_base)
        self.warmup_epochs = int(warmup_epochs)
        # self.global_num = 0
        # 运行时设定
        self.sigma: float = self.n_sigma
        # 统计量初始化（放在 CPU，首次使用时迁移到输入 device）
        # 新增控制开关
        self.use_dynamic_threshold = use_dynamic_threshold
        self.use_soft_weighting = use_soft_weighting
        self.enable_monitor = bool(enable_monitor)

        # 监控指标累加器 (per-epoch 重置)
        self._monitor = {
            'pred_ch_pixels': 0.0,
            'selected_ch_pixels': 0.0,
            'selected_unch_pixels': 0.0,
            'total_pixels': 0.0,
            'ch_count': 0.0,
            'unch_count': 0.0,
            '_epoch': -1,
        }
        if not self.per_class:
            self.prob_max_mu = torch.tensor(0.5, dtype = torch.float32)  # scalar
            self.prob_max_var = torch.tensor(0.05, dtype = torch.float32)  # scalar
        else:
            if self.num_classes == 2:
                # 【关键修改】针对二分类变化检测的非对称初始化
                # 类别0 (背景): 简单样本，初始期望设为 0.9，保持高纯度
                # 类别1 (变化): 困难样本，初始期望设为 0.2，大幅降低门槛，确保早期能召回 0.3~0.4 的弱预测
                self.prob_max_mu = torch.tensor([0.9, 0.5], dtype=torch.float32)
            else:
                # 其他多分类情况，保持默认
                self.prob_max_mu = torch.full((self.num_classes,), 1.0 / self.num_classes, dtype = torch.float32)

            self.prob_max_var = torch.full((self.num_classes,), 0.05, dtype = torch.float32)

        # per-class 上下界 (仅在 num_classes==2 && per_class 时生效)
        # 顺序: [背景, 变化]
        if self.per_class and self.num_classes == 2:
            self.min_thr_per_class = torch.tensor([0.50, 0.50], dtype=torch.float32)
            self.max_thr_per_class = torch.tensor([0.95, 0.95], dtype=torch.float32)

    @torch.no_grad()
    def _ensure_device_dtype(self, ref: Tensor):
        """将统计量迁移到与 ref 相同 device/dtype。"""
        if self.prob_max_mu.device != ref.device or self.prob_max_mu.dtype != ref.dtype:
            self.prob_max_mu = self.prob_max_mu.to(device = ref.device, dtype = ref.dtype)
        if self.prob_max_var.device != ref.device or self.prob_max_var.dtype != ref.dtype:
            self.prob_max_var = self.prob_max_var.to(device = ref.device, dtype = ref.dtype)

    @torch.no_grad()
    def _sigma_schedule(self, epoch: int, total_epochs: int):
        """余弦退火：从 self.n_sigma 平滑下降到 0.5。"""
        progress = max(0.0, min(1.0, epoch / max(1, total_epochs)))
        # 线性变换到 [0.5, self.n_sigma]：y = A + B * cos(π * progress)
        A = (self.n_sigma + 0.5) / 2.0
        B = (self.n_sigma - 0.5) / 2.0
        sigma = A + B * math.cos(math.pi * progress)
        self.sigma = float(sigma)

    # def _sigma(self, algorithm, epoch, total_epochs):
    #     progress = epoch / total_epochs
    #     # self.sigma = 0.5 * self.n_sigma * (1 + np.cos(np.pi * progress))  # 随训练轮数降低
    #     # self.sigma = np.clip(self.sigma, 0.25 * self.n_sigma, self.n_sigma)  # 限制范围
    #     self.sigma = 0.5 * self.n_sigma * (1 + torch.cos(torch.tensor(np.pi * progress)))
    #     self.sigma = torch.clamp(self.sigma, 0.25 * self.n_sigma, self.n_sigma)

    @torch.no_grad()
    def _momentum_schedule(self, epoch: int) -> float:
        """前若干 epoch 使用更快 EMA，以加速冷启动。"""
        if epoch < self.warmup_epochs:
            return 0.999  # 更快地贴近数据
        return self.m_base

    @torch.no_grad()
    def _to_probs(self, logits_or_probs: Tensor, is_logits: bool) -> Tensor:
        """将输入统一为概率分布 [B,C,H,W]。"""
        x = logits_or_probs
        if is_logits:
            if x.shape[1] == 1:
                p1 = torch.sigmoid(x)  # 变化/前景概率
                p0 = 1.0 - p1  # 背景概率
                probs = torch.cat([p0, p1], dim = 1)  # [B,2,H,W]
            else:
                probs = torch.softmax(x, dim = 1)
        else:
            probs = x
            # 若单通道概率，扩展为二分类
            if probs.shape[1] == 1 and self.num_classes == 2:
                p1 = torch.clamp(probs, 0.0, 1.0)
                p0 = 1.0 - p1
                probs = torch.cat([p0, p1], dim = 1)
        return probs

    @torch.no_grad()
    def _dynamic_threshold_bounds(
        self, epoch: int, cls: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        统一计算阈值上下界（全局 / per-class 入口一致）。

        Args:
            epoch: 当前训练轮数。
            cls:   类别索引。None 表示使用全局 [min_thr_base, max_thr_base]；
                   int  表示 per-class 模式 (num_classes==2 && per_class) 时使用
                   [min_thr_per_class[cls], max_thr_per_class[cls]]。

        Returns:
            (min_thr, max_thr): 与统计量同 dtype/device 的 scalar Tensor。

        行为:
          - warmup 阶段 (epoch < warmup_epochs) 下限放宽到 0.05，避免抑制有效伪标签。
          - per_class + num_classes==2 + cls!=None 时取 per-class 上下界；
            其他情况一律取全局基础上下界。
        """
        # 1. 选择基础上下界
        if self.per_class and self.num_classes == 2 and cls is not None:
            min_thr = self.min_thr_per_class[cls]
            max_thr = self.max_thr_per_class[cls]
        else:
            min_thr = torch.as_tensor(self.min_thr_base, dtype = torch.float32)
            max_thr = torch.as_tensor(self.max_thr_base, dtype = torch.float32)

        # 2. warmup 阶段放宽下限
        if epoch < self.warmup_epochs:
            min_thr = torch.minimum(
                min_thr, torch.as_tensor(0.05, dtype = min_thr.dtype)
            )

        return min_thr, max_thr

    @torch.no_grad()
    def update(self, epoch: int, probs_x_ulb: Tensor):
        """
        按论文式 (3)/(5) 更新统计量：优先使用 H，样本不足时回退到 S={p>0.5}。
        H 至少含两个样本时 beta=1，否则 beta=0.1；S 仍不足两个样本则跳过更新。
        prob_max_var 存储方差，阈值和软加权计算时按需转换为标准差。
        """
        B, C, H, W = probs_x_ulb.shape
        assert C == self.num_classes, f"输入通道数{C}与num_classes={self.num_classes}不匹配"
        self._ensure_device_dtype(probs_x_ulb)

        # 调度
        m = self._momentum_schedule(epoch)
        min_thr, max_thr = self._dynamic_threshold_bounds(epoch)

        # -------------------- 统计量更新 --------------------
        if not self.per_class:
            # [全局统计模式] 逻辑保持微调，也可应用类似的保护机制
            max_probs = probs_x_ulb.max(dim=1).values  # [B,H,W]
            if epoch < self.warmup_epochs:
                high_conf_mask = (max_probs > 0.5)
            else:
                mu = torch.clamp(self.prob_max_mu, min=self.min_mu)
                var = torch.clamp(self.prob_max_var, min=self.eps)
                dynamic_threshold = torch.clamp(mu - self.sigma * torch.sqrt(var), min=min_thr, max=max_thr)
                high_conf_mask = (max_probs > dynamic_threshold)

            probs_high = max_probs[high_conf_mask]
            sample_count = probs_high.numel()

            if sample_count > 1:
                samples = probs_high
                beta = 1.0
            else:
                samples = max_probs[max_probs > 0.5]
                beta = 0.1
            if samples.numel() < 2:
                return

            mu_new = samples.mean()
            var_new = torch.var(samples, unbiased=True)

            self.prob_max_mu = m * self.prob_max_mu + beta * (1 - m) * mu_new
            self.prob_max_var = m * self.prob_max_var + beta * (1 - m) * var_new

        else:
            # [按类别统计模式] (变化检测主要用这个，修复重点在这里)

            max_cls = probs_x_ulb.argmax(dim=1)  # [B,H,W]

            # 监控: per-epoch 重置 + 累加
            if self.enable_monitor:
                total_pixels = float(max_cls.numel())
                if self._monitor['_epoch'] != epoch:
                    self._monitor = {
                        'pred_ch_pixels': 0.0,
                        'selected_ch_pixels': 0.0,
                        'selected_unch_pixels': 0.0,
                        'total_pixels': 0.0,
                        'ch_count': 0.0,
                        'unch_count': 0.0,
                        '_epoch': epoch,
                    }
                self._monitor['pred_ch_pixels'] += float((max_cls == 1).sum().item())
                self._monitor['total_pixels'] += total_pixels

            for cls in range(self.num_classes):
                # 提取当前类别的概率图 [B, H, W]
                cls_probs = probs_x_ulb[:, cls, :, :]
                is_dom = (max_cls == cls)

                if epoch < self.warmup_epochs:
                    high_conf_mask = (cls_probs > 0.5) & is_dom
                else:
                    mu = torch.clamp(self.prob_max_mu[cls], min=self.min_mu)
                    var = torch.clamp(self.prob_max_var[cls], min=self.eps)
                    # 统一通过 _dynamic_threshold_bounds 取上下界 (per-class)
                    min_thr_cls, max_thr_cls = self._dynamic_threshold_bounds(epoch, cls=cls)
                    min_thr_cls = min_thr_cls.to(device=cls_probs.device, dtype=cls_probs.dtype)
                    max_thr_cls = max_thr_cls.to(device=cls_probs.device, dtype=cls_probs.dtype)
                    dynamic_threshold = torch.clamp(mu - self.sigma * torch.sqrt(var), min=min_thr_cls, max=max_thr_cls)
                    high_conf_mask = (cls_probs > dynamic_threshold) & is_dom

                probs_cls_high = cls_probs[high_conf_mask]
                sample_count = probs_cls_high.numel()

                # 监控: 累加被当前类别阈值选中的像素数
                if self.enable_monitor:
                    if cls == 1:
                        self._monitor['selected_ch_pixels'] += float(high_conf_mask.sum().item())
                        self._monitor['ch_count'] += float(sample_count)
                    else:
                        self._monitor['selected_unch_pixels'] += float(high_conf_mask.sum().item())
                        self._monitor['unch_count'] += float(sample_count)

                if sample_count > 1:
                    samples = probs_cls_high
                    beta = 1.0
                else:
                    samples = cls_probs[(cls_probs > 0.5) & is_dom]
                    beta = 0.1
                # 无偏方差至少需要两个样本；否则保留已有统计量。
                if samples.numel() < 2:
                    continue

                mu_new = samples.mean()
                var_new = torch.var(samples, unbiased=True)
                # 严格对应式 (5)：beta 仅衰减本批次贡献，旧统计量系数仍为 m。
                self.prob_max_mu[cls] = m * self.prob_max_mu[cls] + beta * (1 - m) * mu_new
                self.prob_max_var[cls] = m * self.prob_max_var[cls] + beta * (1 - m) * var_new

    @torch.no_grad()
    def masking(self, algorithm, epoch: int, logits_x_ulb: Tensor, is_logits: bool = True, is_update: bool = True,
                total_epochs: int = 105) -> Tensor:
        """
        计算权重掩码 [B,H,W]。
        已修改以支持消融实验：
          - use_dynamic_threshold=False: 使用固定阈值 0.95
          - use_soft_weighting=False: 使用硬截断 (0/1)
        """
        self._sigma_schedule(epoch, total_epochs)
        probs = self._to_probs(logits_x_ulb, is_logits=is_logits)
        
        # 即使在消融模式下，我们也通常继续更新统计量，
        # 除非你想完全冻结统计量（通常消融实验只控制使用侧，不控制更新侧，以保持环境一致）
        if is_update:
            self.update(epoch, probs)

        # -------------------- 权重计算 --------------------
        B, C, H, W = probs.shape
        max_probs, max_idx = probs.permute(0, 2, 3, 1).reshape(-1, C).max(dim=1)  # [N], N=BHW

        self._ensure_device_dtype(probs)
        mu = torch.clamp(self.prob_max_mu, min=self.min_mu)
        var = torch.clamp(self.prob_max_var, min=self.eps)

        if not self.per_class:
            mu = mu.expand_as(max_probs)
            var = var.expand_as(max_probs)
        else:
            mu = mu[max_idx]
            var = var[max_idx]

        min_thr, max_thr = self._dynamic_threshold_bounds(epoch)

        # >>> 修改 1: 动态阈值消融 >>>
        if self.use_dynamic_threshold:
            # 完整版: 动态计算阈值
            raw_thr = mu - self.sigma * torch.sqrt(var)
            if self.per_class and self.num_classes == 2:
                # 统一通过 _dynamic_threshold_bounds 取 per-class 上下界
                # warmup 阶段 (_dynamic_threshold_bounds 内部处理) 下限放宽到 0.05
                min_thr_per_class = self.min_thr_per_class.to(
                    device=max_probs.device, dtype=max_probs.dtype
                )
                max_thr_per_class = self.max_thr_per_class.to(
                    device=max_probs.device, dtype=max_probs.dtype
                )
                min_thr_pc = min_thr_per_class[max_idx]
                max_thr_pc = max_thr_per_class[max_idx]
                # warmup 时整体放宽下限，与 _dynamic_threshold_bounds 行为保持一致
                if epoch < self.warmup_epochs:
                    min_thr_pc = torch.clamp(min_thr_pc, max=0.05)
                dynamic_threshold = torch.clamp(raw_thr, min=min_thr_pc, max=max_thr_pc)
            else:
                dynamic_threshold = torch.clamp(raw_thr, min=min_thr, max=max_thr)
        else:
            # 消融版 (NoASPRclass): 使用固定阈值 (例如 0.95)
            # 注意: 这里的固定阈值需要根据你的数据分布选择一个"合理"的值，0.95 是 SSCD 常用的高阈值
            fixed_thr_val = 0.95
            dynamic_threshold = torch.full_like(max_probs, fixed_thr_val)
        # <<< 修改结束 <<<

        # >>> 修改 2: 软加权消融 >>>
        if self.use_soft_weighting:
            # 完整版: 高斯软加权
            # 仅惩罚低于阈值的部分
            clamped_diff = torch.clamp(max_probs - dynamic_threshold, max=0.0) 
            denom = 2.0 * torch.clamp(var, min=self.eps) * max(self.sigma * self.sigma, self.eps)
            weights = torch.exp(-(clamped_diff * clamped_diff) / denom)
        else:
            # 消融版 (NoASPRsoft): 硬截断
            # 大于阈值为 1，小于阈值为 0
            # 这退化为类似 FixMatch 的逻辑
            weights = (max_probs >= dynamic_threshold).float()
        # <<< 修改结束 <<<

        return weights.reshape(B, H, W)

    @torch.no_grad()
    def get_thresholds(self, algorithm, epoch: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """返回: thresholds, sigma, mu, var。"""
        var = torch.clamp(self.prob_max_var, min = self.eps)
        mu = torch.clamp(self.prob_max_mu, min = self.min_mu)
        sigma_t = torch.as_tensor(self.sigma, device = mu.device, dtype = mu.dtype)
        
        # >>> 修改开始 >>>
        # 如果禁用了动态阈值，直接返回固定的 0.95 (或者你设定的那个固定值)
        if not self.use_dynamic_threshold:
            fixed_val = 0.95 # 确保这里和 masking 里用的固定值一致
            if self.per_class:
                thr = torch.full_like(mu, fixed_val)
            else:
                thr = torch.tensor([fixed_val], device=mu.device)
            return thr, sigma_t, mu, var
        # <<< 修改结束 <<<

        min_thr, max_thr = self._dynamic_threshold_bounds(epoch)
        if self.per_class and self.num_classes == 2:
            # 统一通过 _dynamic_threshold_bounds 取 per-class 上下界
            # 这里需要按类别逐个查询，再 stack 成张量
            min_thr_pc = torch.stack(
                [self._dynamic_threshold_bounds(epoch, cls=c)[0] for c in range(self.num_classes)]
            ).to(device=mu.device, dtype=mu.dtype)
            max_thr_pc = torch.stack(
                [self._dynamic_threshold_bounds(epoch, cls=c)[1] for c in range(self.num_classes)]
            ).to(device=mu.device, dtype=mu.dtype)
            thr = torch.clamp(mu - self.sigma * torch.sqrt(var), min=min_thr_pc, max=max_thr_pc)
        else:
            thr = torch.clamp(mu - self.sigma * torch.sqrt(var), min=min_thr, max=max_thr)
            if not self.per_class:
                thr = thr.view(1)
        return thr, sigma_t, mu, var

    @torch.no_grad()
    def get_monitor_metrics(self, algorithm=None) -> dict:
        """
        返回当前 epoch 累积的类别选择偏差监控指标:
          - pred_ch_ratio:        模型预测为变化类的像素比例
          - selected_ch_ratio:    被变化类阈值选中的像素比例
          - selected_unch_ratio:  被背景类阈值选中的像素比例
          - ch_count:             更新 ch_mu/ch_var 的像素数
          - unch_count:           更新 unch_mu/unch_var 的像素数
          - ch_utilization:       selected_ch_pixels / pred_ch_pixels (变化预测被采纳的比例)
        """
        m = self._monitor
        total = max(m['total_pixels'], 1.0)
        return {
            'pred_ch_ratio': m['pred_ch_pixels'] / total,
            'selected_ch_ratio': m['selected_ch_pixels'] / total,
            'selected_unch_ratio': m['selected_unch_pixels'] / total,
            'ch_count': m['ch_count'],
            'unch_count': m['unch_count'],
            'ch_utilization': m['selected_ch_pixels'] / max(m['pred_ch_pixels'], 1.0),
        }

    # # 初始化（二分类场景）
# hook = GaussianDynamicThresholdHook(
#     num_classes = 2,
#     n_sigma = 2,
#     momentum = 0.999,
#     per_class = True
# )
#
# # 模拟输入（batch=4, 256x256图像）
# logits = torch.randn(4, 2, 256, 256).cuda()
#
# # 计算权重掩码
# mask = hook.masking(None, logits)  # 输出维度 [4,256,256]
#
# # 获取当前阈值
# thresholds = hook.get_thresholds()  # tensor([mu0-2σ0, mu1-2σ1], device='cuda:0')
# print(thresholds)
