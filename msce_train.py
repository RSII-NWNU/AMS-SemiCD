import contextlib
from collections import OrderedDict
import argparse
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
from model import SemiModel
import time
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts
from utils import *
from msce_visual import ChangeDetectionVisualizer

start = time.time()

class WarmupCosineSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, T_0, T_mult = 1, eta_min = 0, last_epoch = -1):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self._base_scheduler = None
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 线性预热阶段
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / (self.warmup_epochs + 1)
                    for base_lr in self.base_lrs]
        # 余弦退火阶段
        if self._base_scheduler is None:
            self._init_base_scheduler()
        return self._base_scheduler.get_last_lr()

    def _init_base_scheduler(self):
        """延迟初始化以同步epoch计数"""
        self._base_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0 = self.T_0,
            T_mult = self.T_mult,
            eta_min = self.eta_min,
            last_epoch = max(-1, self.last_epoch - self.warmup_epochs)  # 关键修复点
        )

    def step(self, epoch = None):
        # 调用父类方法自动处理epoch参数
        super().step(epoch)  # 此时self.last_epoch已更新

        # 使用self.last_epoch而非传入的epoch参数
        if self._base_scheduler is not None and self.last_epoch >= self.warmup_epochs:
            self._base_scheduler.step(self.last_epoch - self.warmup_epochs)


class WarmupCosineScheduleByIteration(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_0, T_mult = 1, eta_min = 1e-6, last_epoch = -1):
        self.warmup_steps = warmup_steps
        self.T_0 = T_0  # 周期长度（迭代次数）
        self.T_mult = T_mult  # 周期倍增系数
        self.eta_min = eta_min  # 最小学习率
        self._base_scheduler = None
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 线性预热阶段
        if self.last_epoch < self.warmup_steps:
            progress = (self.last_epoch + 1) / (self.warmup_steps + 1)
            return [base_lr * progress for base_lr in self.base_lrs]
        # 余弦退火阶段
        if self._base_scheduler is None:
            self._init_base_scheduler()
        return self._base_scheduler.get_last_lr()

    def _init_base_scheduler(self):
        """在首次进入余弦阶段时初始化基础调度器"""
        base_last_epoch = self.last_epoch - self.warmup_steps - 1
        self._base_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0 = self.T_0,
            T_mult = self.T_mult,
            eta_min = self.eta_min,
            last_epoch = max(-1, base_last_epoch)  # 确保从正确位置开始
        )

    def step(self, epoch = None):
        # 强制按迭代更新，禁止传入epoch
        super().step(epoch = None)
        # 仅在进入余弦阶段后更新基础调度器
        if self._base_scheduler is not None:
            self._base_scheduler.step()


# 设置种子和参数，确保整个深度学习实验过程中使用的所有随机操作都是可重复的，保证实验结果的一致性。
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    # 获取主进程的初始种子（需在主进程中提前设置）
    base_seed = torch.initial_seed() % 2 ** 32
    # 为每个 worker 生成唯一但确定的种子
    worker_seed = base_seed + worker_id
    # 设置 Python、NumPy、PyTorch 的随机种子
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    # 如果是 CUDA 环境，设置 CUDA 种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)


class ConstantScheduler:
    """用于可视化Loader的伪调度器，永远返回0（禁用Mosaic）"""
    def get_ratio(self):
        return 0.0

class AMS(Base):
    """
        SoftMatch algorithm (https://arxiv.org/abs/2301.10921)).
        SemiReward algorithm (https://arxiv.org/abs/2310.03013).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger.py
            - logger.py (`logging.Logger`):
                logger.py to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ema_p (`float`):
                exponential moving average of probability update
        """

    def __init__(self, model, ema_model, args):
        super().__init__()
        self.init(T = args.T, hard_label = args.use_hard_label, dist_align = args.dist_uniform,
                  dist_uniform = args.dist_uniform,
                  ema_p = args.ema_p, n_sigma = args.n_sigma, per_class = args.per_class,
                  it = args.it, epoch = args.epoch,
                  align_ratio = args.align_ratio,
                  save_path = args.save_path, use_aspr = args.use_aspr)
        self.args = args  # 保存 opt 的引用，而非字段值
        self.num_classes = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_timing = args.start_timing

        self.lambda_u = args.lambda_u

        # 跟踪训练过程中最大的奖励值，用于确保选取最自信的伪标签
        self.max_reward = -float('inf')
        self.model = model
        self.ema_model = ema_model
        self.consistency_loss = ConsistencyLoss()
        self.criterion_comp = TripletContrastiveLoss(0.1)  # 对比损失
        self.hooks_dict = OrderedDict()
        self.set_hooks()

        self.visualizer = ChangeDetectionVisualizer(dataset_name=args.data_name)
        self.vis_epochs = []

    def init(self, T, hard_label = True, dist_align = True, dist_uniform = False, ema_p = 0.999, n_sigma = 2,
             per_class = False, it = 1, epoch = 100, align_ratio = 0.3,
             save_path = None, use_aspr = True):
        self.T = T  # 伪标签软化过程中使用
        self.use_hard_label = hard_label  # 是否使用硬标签
        self.dist_align = dist_align  # 分布对齐策略
        self.dist_uniform = dist_uniform
        self.ema_p = ema_p  # 指数滑动平均的参数
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.epoch = epoch
        self.it = it
        self.align_ratio = align_ratio
        self.save_path = save_path
        self.use_aspr = use_aspr

    def run_visualization(self, epoch, vis_loader, logger):
        vis_save_dir = os.path.join(self.save_path, 'vis_pseudo_evolution', f'epoch_{epoch}')
        os.makedirs(vis_save_dir, exist_ok=True)
        
        # GT 保存路径
        gt_save_dir = os.path.join(self.save_path, 'vis_pseudo_evolution', 'ground_truth')
        if epoch == self.vis_epochs[0]:
            os.makedirs(gt_save_dir, exist_ok=True)
        
        logger.info(f"[Visualization] Epoch {epoch}...")

        # 1. 切换模式并清理 Mamba 状态/显存
        self.ema_model.eval()
        torch.cuda.empty_cache() 
        
        loop = tqdm(vis_loader, desc=f"Vis Epoch {epoch}", leave=False)
            
        for i, batch_data in enumerate(loop):
            try:
                A_w = batch_data[0].to(self.device, non_blocking=True)
                B_w = batch_data[1].to(self.device, non_blocking=True)
                y_lb = batch_data[4].to(self.device, non_blocking=True)
                filenames = batch_data[6]
                
                with torch.no_grad():
                    # 推理
                    pseudo = self.ema_model(A_w, B_w)
                    pseudo_preds = pseudo[1]
                    pseudo_probs = torch.sigmoid(pseudo_preds.detach())

                    mask_pred = None
                    if self.args.use_aspr:
                        mask_pred = self.call_hook("masking", "MaskingHook", epoch=epoch,
                                                   logits_x_ulb=pseudo_probs,
                                                   is_logits=False, is_update=False, total_epochs=self.args.epoch)

                    # --- [关键修改] 维度与数据清理 ---
                    
                    # 确保 y_lb (GT) 是二值的 (处理 Resize 带来的插值噪声)
                    # 大于 0.5 设为 1，否则为 0
                    y_lb = (y_lb > 0.5).float()
                    
                    if y_lb.dim() == 4: y_lb_sq = y_lb.squeeze(1)
                    else: y_lb_sq = y_lb

                    # ---------------------------------------
                    # 1. 保存 Raw Heatmap
                    # ---------------------------------------
                    vis_probs_raw = pseudo_probs.clone()
                    if vis_probs_raw.dim() == 4:
                        if vis_probs_raw.shape[1] == 1: vis_probs_raw = vis_probs_raw.squeeze(1)
                        elif vis_probs_raw.shape[1] == 2: vis_probs_raw = vis_probs_raw[:, 1, :, :] 
                        else: vis_probs_raw = vis_probs_raw.mean(dim=1)
                    
                    self.visualizer.save_heatmap_on_masks(
                        prob_maps=vis_probs_raw,
                        masks=y_lb_sq, 
                        ids=filenames, 
                        save_dir=vis_save_dir,
                        suffix='_raw'
                    )

                    # ---------------------------------------
                    # 2. 保存 Refined Heatmap
                    # ---------------------------------------
                    if mask_pred is not None:
                        # 1. [关键] 先 detach，切断梯度，防止爆显存
                        # 同时转为 float (虽然一般已经是 float，但为了 avg_pool 安全)
                        mask_for_vis = mask_pred.detach().float()
                    
                        # 2. [关键] 确保是 4D 张量 [B, C, H, W] 以适配 avg_pool2d
                        if mask_for_vis.dim() == 3:
                            mask_for_vis = mask_for_vis.unsqueeze(1)
                        
                        # 3. 执行平滑 (Kernel=3 或 5, padding 保持尺寸不变)
                        # 这里的 3x3 平滑足去除“锯齿线”，同时保留大部分轮廓
                        mask_pred_soft = torch.nn.functional.avg_pool2d(
                            mask_for_vis, 
                            kernel_size=3, 
                            stride=1, 
                            padding=1
                        )
                    
                        # 4. 计算 Refined 图
                        # [安全建议] 确保 pseudo_probs 也是 4D，以配合 mask_pred_soft
                        if pseudo_probs.dim() == 3:
                            pseudo_probs_safe = pseudo_probs.unsqueeze(1)
                        else:
                            pseudo_probs_safe = pseudo_probs
                        vis_probs_refined = pseudo_probs_safe * mask_pred_soft
                        
                        if vis_probs_refined.dim() == 4:
                            if vis_probs_refined.shape[1] == 1: vis_probs_refined = vis_probs_refined.squeeze(1)
                            elif vis_probs_refined.shape[1] == 2: vis_probs_refined = vis_probs_refined[:, 1, :, :] 
                            else: vis_probs_refined = vis_probs_refined.mean(dim=1)

                        self.visualizer.save_heatmap_on_masks(
                            prob_maps=vis_probs_refined,
                            masks=y_lb_sq, 
                            ids=filenames, 
                            save_dir=vis_save_dir,
                            suffix='_refined'
                        )
                    
                    # 保存一次 GT 即可
                    if self.vis_epochs and epoch == self.vis_epochs[0]:
                        self.visualizer.save_single_prediction(
                            out_data=y_lb_sq, # 使用处理后的二值化 mask
                            id=filenames,
                            save_dir=gt_save_dir,
                            filename_suffix='_gt'
                        )

            except Exception as e:
                logger.error(f"Batch {i} Vis Error: {e}")
                continue
        
        self.ema_model.train()
        
    @contextlib.contextmanager
    def perf_scope(self, epoch, device, info_bar = None):
        """
        统一的性能分析与 GPU 监控上下文。
        - 开关由 self.args.profile / self.args.monitor_gpu 控制（均可缺省，自动取默认）。
        - 进入时自动初始化 profiler（按调度写入 TensorBoard）、重置本轮峰值显存。
        - 迭代内调用 tick(step_idx) 即可推进 profiler 并按间隔刷新 GPU 信息栏描述。
        - 退出时自动回收资源。

        用法（示例）：
            with self.perf_scope(epoch, device, info3) as tick:
                for i, ... in enumerate(train_loader):
                    ...
                    tick(i)  # 推进 profiler 并刷新 GPU 文本（按间隔）
        """
        import os
        import torch
        import contextlib as _ctx
        # 1) 解析配置（提供健壮的默认值，未在 argparse 中声明也可运行）
        monitor_gpu = bool(getattr(self.args, 'monitor_gpu', False))
        gpu_log_every = int(getattr(self.args, 'gpu_log_every', 10))

        # profiler 参数
        try:
            import torch.profiler as prof
            _prof_avail = True
        except Exception:
            prof = None
            _prof_avail = False
        profile_on = bool(getattr(self.args, 'profile', False)) and _prof_avail
        profile_dir = str(getattr(self.args, 'profile_dir', './runs/profile'))
        wait = int(getattr(self.args, 'profile_wait', 2))
        warmup = int(getattr(self.args, 'profile_warmup', 2))
        active = int(getattr(self.args, 'profile_active', 6))
        repeat = int(getattr(self.args, 'profile_repeat', 1))
        record_shapes = bool(getattr(self.args, 'profile_record_shapes', False))
        profile_memory = bool(getattr(self.args, 'profile_memory', False))
        with_stack = bool(getattr(self.args, 'profile_stack', False))

        # 2) NVML 句柄（可选）
        pynvml = None
        nvml_handle = None
        gpu_index = 0
        if monitor_gpu and torch.cuda.is_available():
            try:
                import pynvml as _pynvml
                pynvml = _pynvml
                pynvml.nvmlInit()
                gpu_id_str = str(getattr(self.args, 'gpu_id', '0'))
                gpu_index = int(gpu_id_str.split(',')[0])
                nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                pynvml = None
                nvml_handle = None  # 回退到 torch.cuda 统计

        # 3) Profiler 上下文（可选）
        if profile_on:
            activities = [prof.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(prof.ProfilerActivity.CUDA)
            epoch_dir = os.path.join(profile_dir, f'epoch_{epoch}')
            os.makedirs(epoch_dir, exist_ok = True)
            prof_ctx = prof.profile(
                activities = activities,
                schedule = prof.schedule(wait = wait, warmup = warmup, active = active, repeat = repeat),
                on_trace_ready = prof.tensorboard_trace_handler(epoch_dir),
                record_shapes = record_shapes,
                profile_memory = profile_memory,
                with_stack = with_stack,
                with_modules = True,
            )
        else:
            prof_ctx = _ctx.nullcontext()

        # 4) 重置本轮峰值显存（仅在启用 profiler 时执行）
        if profile_on and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass

        # 5) 进入统一上下文，提供 tick(step_idx) 回调
        with prof_ctx as p:

            def _gpu_text():
                try:
                    if nvml_handle is not None:
                        util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle).gpu
                        mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                        used_gb = mem.used / (1024 ** 3)
                        total_gb = mem.total / (1024 ** 3)
                        if torch.cuda.is_available():
                            alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                            reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                        else:
                            alloc = reserved = peak = 0.0
                        return f"GPU{gpu_index} util:{util:3d}% mem:{used_gb:.2f}/{total_gb:.2f}G | alloc:{alloc:.2f}G resv:{reserved:.2f}G max:{peak:.2f}G"
                    elif torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(device)
                        total_gb = props.total_memory / (1024 ** 3)
                        alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                        return f"GPU mem:{alloc:.2f}/{total_gb:.2f}G | resv:{reserved:.2f}G max:{peak:.2f}G"
                    else:
                        return "GPU: CPU 模式"
                except Exception:
                    return "GPU: 统计失败"

            last_gpu_update = -1

            def tick(step_idx: int):
                nonlocal last_gpu_update
                # GPU 信息栏（按步长刷新）
                if monitor_gpu and info_bar is not None and torch.cuda.is_available():
                    if last_gpu_update == -1 or (step_idx - last_gpu_update) >= max(1, gpu_log_every):
                        info_bar.set_description_str(_gpu_text())
                        last_gpu_update = step_idx
                # 推进 profiler 调度
                if profile_on:
                    p.step()

            try:
                yield tick
            finally:
                # 6) 资源回收
                try:
                    if pynvml is not None and nvml_handle is not None:
                        pynvml.nvmlShutdown()
                except Exception:
                    pass

    def set_hooks(self):
        self.pseudo_strategy = getattr(self.args, 'pseudo_strategy', 'aspr')

        if self.pseudo_strategy == 'flexmatch':
            masking_hook = FlexMatchStyleHook(num_classes=self.num_classes, p_cutoff=0.95)
        elif self.pseudo_strategy == 'freematch':
            masking_hook = FreeMatchStyleHook(
                num_classes=self.num_classes, momentum=self.ema_p
            )
        elif self.pseudo_strategy == 'softmatch':
            masking_hook = SoftMatchStyleHook(
                num_classes=self.num_classes, n_sigma=self.n_sigma, momentum=self.ema_p
            )
        elif self.pseudo_strategy != 'aspr':
            raise ValueError(f'不支持的 pseudo-label strategy: {self.pseudo_strategy}')
        else:
            masking_hook = None

        # 默认开启所有功能
        use_dynamic_threshold = True
        use_soft_weighting = True

        # 注册钩子
        if masking_hook is None:
            masking_hook = AdaptiveStatisticalPseudoLabelRefinement(
                num_classes = self.num_classes, n_sigma = self.n_sigma, momentum = self.ema_p,
                per_class = self.per_class,
                use_dynamic_threshold = use_dynamic_threshold,
                use_soft_weighting = use_soft_weighting,
                enable_monitor = self.args.enable_aspr_monitor,
            )
        self.register_hook(masking_hook, "MaskingHook")
    def update_ema_sync(self, student, teacher, alpha):
        with torch.no_grad():
            # 参数更新（直接遍历参数对）
            for s_param, t_param in zip(student.parameters(), teacher.parameters()):
                t_param.mul_(alpha).add_(s_param.data, alpha = 1 - alpha)

            # Buffer同步
            for s_buffer, t_buffer in zip(student.buffers(), teacher.buffers()):
                t_buffer.data.copy_(s_buffer.data)

    def train_step(self, train_loader, vis_loader, val_loader, Eva_train, Eva_val, Eva_val2,
                   vis, criterion, optimizer, lr_scheduler, use_ema, num_epochs, it, logger):
        epoch_loss = 0
        self.model.train(True)
        self.ema_model.train(True)
        epoch = it
        length = 0

        logger.info(f"半监督开始轮数：{self.start_timing}, 当前轮数：{epoch}")
        with tqdm(total = len(train_loader), desc = f'Epoch {epoch}/{num_epochs}', unit = 'bat',
                  position = 0, leave = True) as pbar, \
                tqdm(total = 0, position = 1, bar_format = '{desc}', leave = True) as info1, \
                tqdm(total = 0, position = 2, bar_format = '{desc}', leave = True) as info2, \
                tqdm(total = 0, position = 3, bar_format = '{desc}', leave = True) as info3, \
                self.perf_scope(epoch, device, info3) as tick:
            total_loss = torch.tensor(0.0, device = device)
            # 正确解包包含弱强增强对的6元组
            for i, (A_w, B_w, A_s, B_s, mask, with_label, filenames) in enumerate(train_loader):
                # ----------------- 数据准备 -----------------
                A_w, B_w = A_w.to(device, non_blocking = True), B_w.to(device, non_blocking = True)
                A_s, B_s = A_s.to(device, non_blocking = True), B_s.to(device, non_blocking = True)
                y_lb = mask.to(device, non_blocking = True)  # Ground Truth标签
                with_label = with_label.to(device, non_blocking = True)

                # ----------------- 梯度重置 -----------------
                optimizer.zero_grad(set_to_none = True)

                # 初始化损失
                loss_su = torch.tensor(0.0, device = device)
                loss_semi = torch.tensor(0.0, device = device)

                # ----------------- 监督损失计算 (对有标签数据) -----------------
                preds_sup = None  # 初始化，用于后续指标计算
                if with_label.any():
                    # Student模型处理有标签数据的强增强版本
                    preds_sup = self.model(A_s[with_label], B_s[with_label])
                    loss_su = criterion(preds_sup[1], y_lb[with_label])
                    if not use_ema and self.use_aspr and self.pseudo_strategy == 'aspr':
                        pred_x_lb = torch.sigmoid(preds_sup[1].detach())
                        _ = self.call_hook("masking", "MaskingHook", epoch = epoch, logits_x_ulb = pred_x_lb,
                                           is_logits = False, total_epochs = num_epochs)

                # ----------------- 半监督损失计算 (对无标签数据) -----------------
                if use_ema and (~with_label).any():
                    # 1. Teacher模型处理无标签数据的弱增强版本，生成伪标签
                    with torch.no_grad():
                        # 使用弱增强视图 (A_w, B_w)
                        pseudo = self.ema_model(A_w[~with_label], B_w[~with_label])
                        pseudo_preds = pseudo[1]
                        pseudo_probs = torch.sigmoid(pseudo_preds.detach())

                        mask_pred = None
                        if self.use_aspr:
                            # 使用伪标签概率计算置信度掩码
                            mask_pred = self.call_hook("masking", "MaskingHook", epoch = epoch,
                                                       logits_x_ulb = pseudo_probs,
                                                       is_logits = False, is_update=True, total_epochs = num_epochs)

                    # 2. Student模型处理无标签数据的强增强版本
                    # 使用强增强视图 (A_s, B_s)
                    preds_unsup = self.model(A_s[~with_label], B_s[~with_label])

                    # 3. 计算一致性损失
                    if self.use_aspr:
                        loss_semi = self.consistency_loss(preds_unsup[1], pseudo_probs, 'ce', mask = mask_pred)
                    else:
                        loss_semi = self.consistency_loss(preds_unsup[1], pseudo_probs, 'ce')

                # ----------------- 损失整合与反向传播 -----------------
                # 只有在至少计算了一种损失的情况下才进行更新
                if not (with_label.any() or (~with_label).any()):
                    pbar.update(1)
                    continue  # 跳过空批次（理论上不应发生）

                loss = loss_su + self.lambda_u * loss_semi

                # 核心修复：只有当 loss 是一个包含计算图的有效值时才进行反向传播
                if loss.requires_grad:
                    if torch.isnan(loss).any():
                        logger.warning(f'批次 {i} 包含 NaN loss, 跳过反向传播！ Loss: {loss.item()}')
                        pbar.update(1)
                        continue

                    loss.backward()
                    optimizer.step()
                else:
                    # 如果 loss 不需要梯度（例如，当批次中只有无标签数据且 use_ema=False 时），
                    # 就直接跳过反向传播和优化器步骤。
                    pbar.update(1)
                    continue

                # ----------------- EMA模型更新 -----------------
                self.update_ema_sync(self.model, self.ema_model, 0.99)

                # ----------------- 训练指标计算与更新 -----------------
                # 只在有标签数据上计算训练指标
                if preds_sup is not None:
                    output = F.sigmoid(preds_sup[1])
                    output = torch.nan_to_num(output, nan = 0.0)
                    output[output >= 0.5] = 1
                    output[output < 0.5] = 0
                    pred = output.squeeze(1)
                    target = y_lb[with_label].squeeze(1)
                    Eva_train.add_batch(target, pred)

                epoch_loss += loss.item()
                length += 1  # 如果需要平均loss，length应该在这里更新

                # 获取当前 pseudo-label strategy 的内部状态用于监控
                thresholds, sigma, prob_max_mu, prob_max_var = self.call_hook("get_thresholds",
                                                                              "MaskingHook", epoch = epoch)

                postfix_line1 = {
                    'Loss': f"{loss.item():.3f}",
                    'SupLoss': f"{loss_su.item():.3f}" if with_label.any() else 0.0,
                    'SemiLoss': f"{loss_semi.item():.3f}" if use_ema else 0.0,
                    'Eva_loss': f"{epoch_loss / length:.3f}",
                }
                if self.use_aspr and self.pseudo_strategy == 'aspr':
                    postfix_line2 = {
                        'unch_thr': f"{thresholds[0].item():.3f}",
                        'ch_thr': f"{thresholds[1].item():.3f}",
                        'sigma': f"{sigma.item():.3f}",
                        'unch_mu': f"{prob_max_mu[0].item():.3f}",
                        'unch_var': f"{prob_max_var[0].item():.3f}",
                        'ch_mu': f"{prob_max_mu[1].item():.3f}",
                        'ch_var': f"{prob_max_var[1].item():.3f}",
                    }
                elif self.use_aspr:
                    strategy_monitor = self.call_hook("get_monitor_metrics", "MaskingHook")
                    postfix_line2 = {
                        'strategy': self.pseudo_strategy,
                        'unch_thr': f"{thresholds[0].item():.3f}",
                        'ch_thr': f"{thresholds[1].item():.3f}",
                        'mean_w': f"{strategy_monitor.get('mean_weight', 0.0):.3f}",
                        'selected': f"{strategy_monitor.get('selected_ratio', 0.0):.3f}",
                    }
                # 两行分别显示
                info1.set_description_str(' '.join([f'{k}:{v}' for k, v in postfix_line1.items()]))
                if self.use_aspr:
                    info2.set_description_str(' '.join([f'{k}:{v}' for k, v in postfix_line2.items()]))

                pbar.update(1)
                tick(i)

        # === [关键步骤] 手动清理显存，防止 OOM ===
        # 删除训练循环中最后遗留的变量，断开计算图
        
        # 1. 必定存在的变量，可以直接删除
        del loss, loss_su, loss_semi
        
        # 2. 可能不存在的变量，需条件删除
        if 'preds_sup' in locals() and preds_sup is not None: 
            del preds_sup
            
        if 'preds_unsup' in locals():
            del preds_unsup
            
        if 'pseudo' in locals(): 
            del pseudo
            
        if 'pseudo_probs' in locals(): 
            del pseudo_probs
            
        if 'mask_pred' in locals(): 
            del mask_pred
        
        # 强制释放 PyTorch 缓存的显存
        torch.cuda.empty_cache()

        # === [新增] 独立运行可视化 ===
        # 此时显存应该已经降下来了，可以安全运行
        if epoch in self.vis_epochs:
            self.run_visualization(epoch, vis_loader, logger)
        # ==========================
        
        # --- (循环结束后的指标汇总和打印代码保持不变) ---
        train_loss = epoch_loss / length  # 使用length计算平均loss

        # ASPR 类别选择偏差监控 (每轮结束输出一次)
        if (self.use_aspr and self.pseudo_strategy == 'aspr'
                and self.args.enable_aspr_monitor):
            mon = self.call_hook("get_monitor_metrics", "MaskingHook")
            if mon is not None:
                logger.info(
                    f'轮数:{epoch}, 总轮数:{num_epochs}, '
                    f'[ASPR-Monitor] pred_ch_ratio:{mon["pred_ch_ratio"]:.4f} '
                    f'selected_ch_ratio:{mon["selected_ch_ratio"]:.4f} '
                    f'selected_unch_ratio:{mon["selected_unch_ratio"]:.4f} '
                    f'ch_util:{mon["ch_utilization"]:.4f} '
                    f'ch_count:{int(mon["ch_count"])} '
                    f'unch_count:{int(mon["unch_count"])}'
                )
        elif self.use_aspr:
            mon = self.call_hook("get_monitor_metrics", "MaskingHook")
            logger.info(
                f'轮数:{epoch}, 总轮数:{num_epochs}, '
                f'[PseudoStrategy-Monitor] strategy:{self.pseudo_strategy} '
                f'mean_weight:{mon.get("mean_weight", 0.0):.4f} '
                f'selected_ratio:{mon.get("selected_ratio", 0.0):.4f}'
            )

        IoU = Eva_train.Intersection_over_Union()[1] if Eva_train.confusion_matrix.sum() > 0 else 0.0
        Pre = Eva_train.Precision()[1] if Eva_train.confusion_matrix.sum() > 0 else 0.0
        Recall = Eva_train.Recall()[1] if Eva_train.confusion_matrix.sum() > 0 else 0.0
        F1 = Eva_train.F1()[1] if Eva_train.confusion_matrix.sum() > 0 else 0.0

        vis.add_scalar(params='IoU_train', value=IoU, epoch=epoch)
        vis.add_scalar(params='Precision_train', value=Pre, epoch=epoch)
        vis.add_scalar(params='Recall_train', value=Recall, epoch=epoch)
        vis.add_scalar(params='F1_train', value=F1, epoch=epoch)
        vis.add_scalar(params='train_loss_train', value=train_loss, epoch=epoch)

        print('轮数/总轮数：[%d/%d],\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
                epoch, num_epochs, IoU, Pre, Recall, F1))
        logger.info(f'轮数:{epoch}, 总轮数:{num_epochs}, [Training] IoU:{IoU:.4f}, Precision:{Pre:.4f}, Recall:{Recall:.4f}, F1:{F1:.4f}, train_loss:{train_loss:.4f}')

        print("开始验证...")
        logger.info('开始验证...')
        self.model.train(False)
        self.model.eval()
        self.ema_model.train(False)
        self.ema_model.eval()
        for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                A = A.to(device, non_blocking=True)
                B = B.to(device, non_blocking=True)
                y_lb = mask.to(device, non_blocking=True)
                
                preds = self.model(A, B)[1]
                output = F.sigmoid(preds)
                output[output >= 0.5] = 1
                output[output < 0.5] = 0
                
                # 【修改】直接传 Tensor 并在 GPU 上完成统计
                pred = output.squeeze(1)
                target = y_lb.squeeze(1)
                Eva_val.add_batch(target, pred)

                preds_ema = self.ema_model(A, B)[1]
                # 【修改】直接传 Tensor，转为 long 类型
                Eva_val2.add_batch(target, (preds_ema.squeeze(1) > 0).long())
                
                length += 1
                """
                    这里到底是存net的参数还是ema_net的参数，都可以，看哪个精度高
                """
        # ----------------- 提取并打印验证集指标 -----------------
        # 【修改】全部加上 [1] 获取类别1（变化区域）的分数
        IoU = Eva_val.Intersection_over_Union()[1]
        Pre = Eva_val.Precision()[1]
        Recall = Eva_val.Recall()[1]
        F1 = Eva_val.F1()[1]

        vis.add_scalar(params='IoU_val', value=IoU, epoch=epoch)
        vis.add_scalar(params='Precision_val', value=Pre, epoch=epoch)
        vis.add_scalar(params='Recall_val', value=Recall, epoch=epoch)
        vis.add_scalar(params='F1_val', value=F1, epoch=epoch)

        IoU_ema = Eva_val2.Intersection_over_Union()[1]
        Pre_ema = Eva_val2.Precision()[1]
        Recall_ema = Eva_val2.Recall()[1]
        F1_ema = Eva_val2.F1()[1]

        vis.add_scalar(params='IoU_ema_val', value=IoU_ema, epoch=epoch)
        vis.add_scalar(params='Precision_ema_val', value=Pre_ema, epoch=epoch)
        vis.add_scalar(params='Recall_ema_val', value=Recall_ema, epoch=epoch)
        vis.add_scalar(params='F1_ema_val', value=F1_ema, epoch=epoch)

        print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU, Pre, Recall, F1))
        logger.info(f'轮数:{epoch}, 总轮数:{num_epochs}, [Validation] IoU:{IoU:.4f}, Precision:{Pre:.4f}, Recall:{Recall:.4f}, F1:{F1:.4f}')

        print('[Ema Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU_ema, Pre_ema, Recall_ema, F1_ema))
        logger.info(f'轮数:{epoch}, 总轮数:{num_epochs}, [Ema Validation] IoU:{IoU_ema:.4f}, Precision:{Pre_ema:.4f}, Recall:{Recall_ema:.4f}, F1:{F1_ema:.4f}')

        new_student_iou = IoU  
        new_teacher_iou = IoU_ema

        save_checkpoint(epoch, model, ema_model, optimizer, lr_scheduler, new_student_iou, new_teacher_iou,
                        self.save_path, self.args, logger, self.use_aspr, self.hooks_dict)

        print('最优学生模型,轮数:%d; Iou :%.4f' % (self.args.best_student_epoch, self.args.best_student_iou))
        print('最优教师模型,轮数:%d; Iou :%.4f' % (self.args.best_teacher_epoch, self.args.best_teacher_iou))
        logger.info(f'最优学生模型,轮数: {self.args.best_student_epoch}, IoU: {self.args.best_student_iou}')
        logger.info(f'最优教师模型,轮数: {self.args.best_teacher_epoch}, IoU: {self.args.best_teacher_iou}')
        vis.close_summary()

if __name__ == '__main__':
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=105, help='epoch number')  # 修改这里！！！
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--lr_head', type = float, default = 2e-4, help = 'learning rate for model head')
    parser.add_argument('--lr_backbone', type = float, default = 2e-5, help = 'learning rate for backbone')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')  # 修改这里！！！
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--train_ratio', type=float, default=0.05,
                        help='Proportion of the labeled images')  # 修改这里！！！
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--seed_id', type=int, default=42, help='seed')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')  # 修改这里！！！
    parser.add_argument('--cpu', action = 'store_true', help = '使用CPU进行推理')
    parser.add_argument('--data_name', type=str, choices=['LEVIR', 'WHU', 'GZ'], default='WHU',
                        help='the test rgb images root')
    parser.add_argument('--model_name', type=str, choices=['SemiModel_ALL'], default='SemiModel_ALL',
                        help='the test rgb images root')
    parser.add_argument(
        '--pseudo_strategy', type=str, default='aspr',
        choices=['aspr', 'flexmatch', 'freematch', 'softmatch'],
        help='Pseudo-label refinement strategy'
    )
    parser.add_argument('--save_path', type=str, default='./output/C2F-SemiCD/')  # 半监督的模型保存路径！！
    parser.add_argument('--ema_p', type=float,
                        default=0.999)  # 动量系数，用于指数衰减更新 prob_max_mu_t 和 prob_max_var_t 的值，SoftMatch截断高斯加权函数相关。
    parser.add_argument('--n_sigma', type=int, default=2) # 控制截断高斯的标准差的超参数，SoftMatch相关。
    
    parser.add_argument('--no_per_class', action='store_false', dest='per_class', default=True, 
                        help='禁用为每个类别分别计算 mu 和 var (默认: 启用)')
    parser.add_argument('--no_res_pretrained', action='store_false', dest='res_pretrained', default=True, 
                        help='不使用 res 预训练权重 (默认: 使用)')
    parser.add_argument('--disable_aspr', action='store_false', dest='use_aspr', default=True,
                        help='在半监督学习中禁用ASPR (默认: 启用)')
    parser.add_argument('--disable_aspr_monitor', action='store_false', dest='enable_aspr_monitor', default=True,
                        help='关闭每轮训练结束时的ASPR类别选择偏差监控 (默认: 启用)')
                        
    parser.add_argument('--use_hard_label', action='store_true', default=False, 
                        help='是否使用硬标签 (默认: 不使用)')
    parser.add_argument('--dist_uniform', action='store_true', default=False, 
                        help='分布对齐策略是否使用 uniform (默认: 不使用)')
    parser.add_argument('--model_load', action='store_true', default=False, 
                        help='是否加载模型 (默认: 不加载)')
    parser.add_argument('--T', type=float, default=0.2)  # 温度参数，用于控制软标签的平滑度。
    parser.add_argument('--align_ratio', type=float, default=0.1)  # 均匀分布对齐的变化概率初始化值
    parser.add_argument('--load_path', type=str,
                        default='./output/C2F-SemiCD/WHU/SemiModel_ALL_WHU_0.05_20260102-124615') 
    parser.add_argument('--soft_path', type=str, default='./output/C2F-SemiCD/')  # soft模型加载路径
    parser.add_argument('--vmamba_weight_path', type=str,
                        default='./model/vmamba/vmambaweight/vssm_small_0229_ckpt_epoch_222.pth')  # 无监督损失比例
    parser.add_argument('--start_timing', type=int, default=5)  # 起始计时
    parser.add_argument('--it', type=int, default=0)  # 训练轮次
    parser.add_argument('--lambda_u', type=float, default=0.2)  # 无监督损失比例


    opt = parser.parse_args()

    seed_everything(opt.seed_id)

    # 设置设备
    if opt.cpu:
        device = torch.device('cpu')
        print("使用CPU进行推理")
    else:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{opt.gpu_id}')
            print(f"使用GPU {opt.gpu_id} 进行推理")
        else:
            device = torch.device('cpu')
            print("CUDA不可用，使用CPU进行推理")

    # ================= [核心修改开始] =================
    # 逻辑：如果是断点续训，直接复用 load_path 作为实验目录；否则生成新目录
    if opt.model_load:
        # 1. 恢复训练模式：直接使用 load_path 作为本次实验的根目录
        # 注意：opt.load_path 应该指向之前的实验文件夹，例如 ./output/C2F-SemiCD/GZ/SemiModel_ALL_GZ_0.1_2024...
        experiment_dir = opt.load_path
        if not os.path.exists(experiment_dir):
            raise FileNotFoundError(f"加载路径不存在: {experiment_dir}")
        print(f"==> 正在恢复训练，结果将继续保存至原路径: {experiment_dir}")
    else:
        # 2. 新训练模式：生成带时间戳的新目录
        # 格式: ./output/C2F-SemiCD/GZ/SemiModel_ALL_GZ_0.1_20260113-150000
        timestr = time.strftime("%Y%m%d-%H%M%S")
        strategy_suffix = '' if opt.pseudo_strategy == 'aspr' else f'_{opt.pseudo_strategy}'
        exp_name = (
            f"{opt.model_name}_{opt.data_name}_{str(opt.train_ratio)}"
            f"{strategy_suffix}_{timestr}"
        )
        # 基础路径 + 数据集名 + 实验名
        experiment_dir = os.path.join(opt.save_path, opt.data_name, exp_name)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        print(f"==> 开始新训练，创建统一实验路径: {experiment_dir}")

    # 将 opt.save_path 更新为这个统一的实验目录
    # 之后所有的权重保存 (save_checkpoint) 都会用到这个更新后的路径
    opt.save_path = experiment_dir

    # [修改点 1] 统一 TensorBoard 指标保存路径
    # 之前是存到 ./runs，现在改到 experiment_dir 下
    vis = visual()
    vis.create_summary(experiment_dir)

    # [修改点 2] 统一日志文件保存路径
    # 之前是存到 ./log，现在改到 experiment_dir 下
    log_name = f'train_{opt.model_name}_{opt.data_name}_{str(opt.train_ratio)}.log'
    log_full_path = os.path.join(experiment_dir, log_name)
    
    # 初始化 logger
    logger = loggering(log_full_path)
    logger.info('PyTorch Version {}\n Experiment Path: {}'.format(torch.__version__, experiment_dir))

    # ================= [输出训练参数] =================
    logger.info('=' * 60)
    logger.info('训练参数配置:')
    logger.info('=' * 60)
    logger.info(f'[数据集] data_name: {opt.data_name}')
    logger.info(f'[数据集] train_ratio: {opt.train_ratio}')
    logger.info(f'[数据集] trainsize: {opt.trainsize}')
    logger.info(f'[模型] model_name: {opt.model_name}')
    logger.info(f'[半监督] pseudo_strategy: {opt.pseudo_strategy}')
    logger.info(f'[模型] vmamba_weight_path: {opt.vmamba_weight_path}')
    logger.info(f'[模型] res_pretrained: {opt.res_pretrained}')
    logger.info(f'[训练] epoch: {opt.epoch}')
    logger.info(f'[训练] batchsize: {opt.batchsize}')
    logger.info(f'[训练] lr_head: {opt.lr_head}')
    logger.info(f'[训练] lr_backbone: {opt.lr_backbone}')
    logger.info(f'[训练] clip: {opt.clip}')
    logger.info(f'[训练] seed_id: {opt.seed_id}')
    logger.info(f'[半监督] lambda_u: {opt.lambda_u}')
    logger.info(f'[半监督] T: {opt.T}')
    logger.info(f'[半监督] ema_p: {opt.ema_p}')
    logger.info(f'[半监督] n_sigma: {opt.n_sigma}')
    logger.info(f'[半监督] start_timing: {opt.start_timing}')
    logger.info(f'[半监督] use_aspr: {opt.use_aspr}')
    logger.info(f'[半监督] use_hard_label: {opt.use_hard_label}')
    logger.info(f'[半监督] dist_uniform: {opt.dist_uniform}')
    logger.info(f'[半监督] per_class: {opt.per_class}')
    logger.info(f'[半监督] align_ratio: {opt.align_ratio}')
    logger.info(f'[GPU] gpu_id: {opt.gpu_id}')
    logger.info(f'[路径] save_path: {opt.save_path}')
    logger.info(f'[路径] model_load: {opt.model_load}')
    if opt.model_load:
        logger.info(f'[路径] load_path: {opt.load_path}')
    logger.info('=' * 60)
    # ================= [核心修改结束] =================

    if opt.data_name == 'LEVIR':
        opt.train_root = '/data/proj/ypc/ams-new/data/LEVIR-CD-256/train/'
        opt.val_root = '/data/proj/ypc/ams-new/data/LEVIR-CD-256/val/'
    elif opt.data_name == 'WHU':
        opt.train_root = '/data/proj/ypc/ams-new/data/WHU-CD-256/train/'
        opt.val_root = '/data/proj/ypc/ams-new/data/WHU-CD-256/val/'
    elif opt.data_name == 'GZ':
        opt.train_root = '/data/proj/ypc/ams-new/data/GZ-CD-256/train/'
        opt.val_root = '/data/proj/ypc/ams-new/data/GZ-CD-256/val/'
    else:
        raise ValueError("仅支持论文数据集：LEVIR、WHU、GZ")

    Eva_train = Evaluator(num_class=2)
    Eva_val = Evaluator(num_class=2)
    Eva_val2 = Evaluator(num_class=2)

    if opt.model_name != 'SemiModel_ALL':
        raise ValueError("仅支持正式模型 SemiModel_ALL")
    model = SemiModel(opt.vmamba_weight_path, opt.res_pretrained).to(device)
    ema_model = SemiModel(None, False).to(device)
    ema_model.load_state_dict(model.state_dict(), strict=True)
    for param in ema_model.parameters():
        param.detach_()

    criterion = nn.BCEWithLogitsLoss().to(device)

    mosaic_scheduler = MosaicScheduler(
        logger,
        total_epochs = opt.epoch,
        pretrain_epochs = 15,  # <--- 在前15轮使用弱增强
        finetune_epochs = 10,  # 微调期将关闭Mosaic
        high_ratio = 0.75,
        low_ratio = 0
    )

    train_loader = get_semi_loader(opt.train_root, opt.batchsize, opt.trainsize, opt.train_ratio,
                                   logger, mosaic_scheduler, worker_init_fn, num_workers=4, shuffle=True,
                                   pin_memory=True)
    val_loader = get_val_loader(opt.val_root, opt.batchsize, opt.trainsize, logger, num_workers=4,
                                shuffle=False, pin_memory=True)

    vis_scheduler = ConstantScheduler()
    vis_dataset = SemiUnsupeDataset(opt.train_root, opt.trainsize, logger, vis_scheduler, sup_indices=[], is_vis=True)
    vis_loader = torch.utils.data.DataLoader(
        dataset=vis_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info("正在为 backbone 和 head 设置不同的学习率...")
    backbone_params = []
    head_params = []
    
    backbone_prefixes = (
        'vmamba.', 
        'firstconv.', 
        'firstbn.', 
        'firstrelu.', 
        'firstmaxpool.', 
        'res1.', 
        'res2.', 
        'res3.', 
        'res4.'
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_backbone = False
        for prefix in backbone_prefixes:
            if name.startswith(prefix):
                is_backbone = True
                break
        
        if is_backbone:
            backbone_params.append(param)
            print(f"[Backbone] {name}") # (可选) 打印以确认分组
        else:
            head_params.append(param)
            print(f"[Head] {name}") # (可选) 打印以确认分组

    param_groups = [
        {'params': head_params, 'lr': opt.lr_head, 'name': 'head'},
        {'params': backbone_params, 'lr': opt.lr_backbone, 'name': 'backbone'}
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=0.0025,
    )

    # 打印优化器信息以确认
    print("优化器设置完成 (精确分组):")
    print(f"  - Head (共 {len(head_params)} 个参数张量) LR: {opt.lr_head}")
    print(f"  - Backbone (共 {len(backbone_params)} 个参数张量) LR: {opt.lr_backbone}")
    logger.info(f"优化器设置完成: Head LR = {opt.lr_head}, Backbone LR = {opt.lr_backbone}")

    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_epochs = 5,  # 预热5个epoch
        T_0 = 50,  # 保持原周期长度
        T_mult = 1,  # 保持原周期倍增
        eta_min = 1e-6  # 保持原最小学习率
    )

    it1 = opt.it
    opt.best_student_epoch = 0
    opt.best_teacher_epoch = 0
    opt.best_student_iou = 0.0
    opt.best_teacher_iou = 0.0
    srsoft = AMS(model=model, ema_model=ema_model, args=opt)
    srsoft.vis_epochs = []
    # 模型恢复
    if opt.model_load:
        # 假设你保存的断点文件名为 latest_checkpoint.pth
        # 如果文件名不同，请修改这里
        resume_path = os.path.join(opt.load_path, 'latest_checkpoint.pth')
        
        if os.path.exists(resume_path):
            logger.info(f"正在加载断点: {resume_path}")
            print(f"正在加载断点: {resume_path}")
            
            # 1. 加载文件 (使用 map_location='cpu' 防止 GPU 显存峰值)
            checkpoint = torch.load(resume_path, weights_only=False, map_location='cpu')

            checkpoint_strategy = checkpoint.get('pseudo_strategy', 'aspr')
            if checkpoint_strategy != opt.pseudo_strategy:
                raise ValueError(
                    f'断点 pseudo_strategy={checkpoint_strategy} 与当前 '
                    f'pseudo_strategy={opt.pseudo_strategy} 不一致'
                )
            
            # 2. 恢复模型权重
            # 注意：保存时处理了 DataParallel，加载时直接 load 即可
            model.load_state_dict(checkpoint['model_state_dict'])
            ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            
            # 3. 恢复优化器和调度器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 4. 恢复 SoftMatch 相关的统计量 (MaskingHook)
            # 对应保存时的: 'prob_max_mu' 和 'prob_max_var'
            if (opt.pseudo_strategy == 'aspr' and 'prob_max_mu' in checkpoint
                    and 'prob_max_var' in checkpoint):
                srsoft.hooks_dict['MaskingHook'].prob_max_mu = checkpoint['prob_max_mu'].to(device)
                srsoft.hooks_dict['MaskingHook'].prob_max_var = checkpoint['prob_max_var'].to(device)
            elif (opt.pseudo_strategy != 'aspr'
                  and 'pseudo_strategy_state' in checkpoint):
                srsoft.hooks_dict['MaskingHook'].load_state_dict(
                    checkpoint['pseudo_strategy_state']
                )
            
            # 5. 恢复最佳指标记录
            opt.best_student_iou = checkpoint.get('best_student_iou', 0.0)
            opt.best_student_epoch = checkpoint.get('best_student_epoch', 0)
            opt.best_teacher_iou = checkpoint.get('best_teacher_iou', 0.0)
            opt.best_teacher_epoch = checkpoint.get('best_teacher_epoch', 0)
            
            # 6. 恢复当前 Epoch (保存的是已完成的 epoch，所以继续训练要 +1)
            # 假设保存时 epoch=10，表示跑完了第10轮，下次应从11开始
            opt.it = checkpoint['epoch'] + 1
            it1 = checkpoint['epoch'] + 1
            
            logger.info(f'模型加载完成，从 Epoch {it1} 继续训练')
            logger.info(f'  恢复的最佳 Student IoU: {opt.best_student_iou:.4f} (Epoch {opt.best_student_epoch})')
            logger.info(f'  恢复的最佳 Teacher IoU: {opt.best_teacher_iou:.4f} (Epoch {opt.best_teacher_epoch})')
            if opt.pseudo_strategy == 'aspr' and 'prob_max_mu' in checkpoint:
                logger.info(
                    f'  已恢复 ASPR 统计量: prob_max_mu={checkpoint["prob_max_mu"]}, '
                    f'prob_max_var={checkpoint["prob_max_var"]}'
                )
            elif opt.pseudo_strategy != 'aspr' and 'pseudo_strategy_state' in checkpoint:
                logger.info(f'  已恢复 {opt.pseudo_strategy} pseudo-label strategy 状态')
            
            # 清理内存
            del checkpoint
            torch.cuda.empty_cache()
            
        else:
            logger.warning(f"未找到断点文件: {resume_path}，将从头开始训练！")
            print(f"警告：未找到断点文件 {resume_path}")

    print("开始训练...")
    print(f'labeled比例 = {opt.train_ratio}, 现在半监督损失函数系数为: {opt.lambda_u}!')
    logger.info('开始训练...')
    logger.info(f'labeled比例 = {opt.train_ratio}, 现在半监督损失函数系数为: {opt.lambda_u}!')
    for it in range(it1, opt.epoch):
        learn_rate = optimizer.param_groups[0]['lr']
        # 格式化输出，保留 6 位小数
        print(f'学习率：{learn_rate:.6f}')
        logger.info(f'学习率：{learn_rate:.6f}')

        logger.info(f'epoch = {it}, total_epoch = {opt.epoch}')
        if opt.it < opt.start_timing:
            use_ema = False
        else:
            use_ema = True

        Eva_train.reset()
        Eva_val.reset()
        Eva_val2.reset()
        try:
            # 训练步骤
            srsoft.train_step(train_loader, vis_loader, val_loader, Eva_train, Eva_val, Eva_val2, vis, criterion, optimizer,
                              lr_scheduler, use_ema, opt.epoch, it, logger)
        except Exception as e:
            logger.error('训练过程中发生错误', exc_info=True)
            logger.error('终止训练')
            break

        lr_scheduler.step()
        mosaic_scheduler.step()
        opt.it += 1

end = time.time()
print('程序训练train的时间为:', end - start)
