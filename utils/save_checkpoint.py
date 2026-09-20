import os
import re
import glob
import torch


def load_state_dict_from_checkpoint(checkpoint_path, model_type='auto'):
    """
    从检查点文件中加载 state_dict，自动识别 key 类型。

    Args:
        checkpoint_path (str): 检查点文件路径。
        model_type (str): 'student' 加载 model_state_dict, 'teacher' 加载 ema_model_state_dict,
                          'auto' 自动检测。

    Returns:
        dict: state_dict，可直接用于 model.load_state_dict()。
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if not isinstance(checkpoint, dict):
        return checkpoint

    if model_type == 'student':
        return checkpoint.get('model_state_dict', checkpoint)
    elif model_type == 'teacher':
        return checkpoint.get('ema_model_state_dict', checkpoint)
    else:
        # 自动检测：优先 ema_model_state_dict，其次 model_state_dict
        if 'ema_model_state_dict' in checkpoint:
            return checkpoint['ema_model_state_dict']
        elif 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        else:
            return checkpoint


def collect_checkpoints(save_path):
    """
    扫描 save_path 下所有可用的检查点文件，返回 [(名称, 路径, 类型), ...] 列表。

    扫描范围：
      - latest_checkpoint.pth
      - best_student_checkpoints/ 目录下所有 .pth
      - best_teacher_checkpoints/ 目录下所有 .pth
      - 根目录下的旧格式 .pth（best_student_checkpoint.pth 等）

    Returns:
        list of tuple: (checkpoint_name, full_path, model_type)
            model_type 为 'student' 或 'teacher'
    """
    checkpoints = []

    # 1. latest_checkpoint.pth
    latest = os.path.join(save_path, 'latest_checkpoint.pth')
    if os.path.exists(latest):
        checkpoints.append(('latest_checkpoint.pth', latest, 'student'))

    # 2. best_student_checkpoints/ 目录
    student_dir = os.path.join(save_path, 'best_student_checkpoints')
    if os.path.isdir(student_dir):
        for f in sorted(glob.glob(os.path.join(student_dir, '*.pth'))):
            name = os.path.basename(f)
            checkpoints.append((name, f, 'student'))

    # 3. best_teacher_checkpoints/ 目录
    teacher_dir = os.path.join(save_path, 'best_teacher_checkpoints')
    if os.path.isdir(teacher_dir):
        for f in sorted(glob.glob(os.path.join(teacher_dir, '*.pth'))):
            name = os.path.basename(f)
            checkpoints.append((name, f, 'teacher'))

    # 4. 根目录下的旧格式文件（兼容）
    for old_name, mtype in [('best_student_checkpoint.pth', 'student'),
                             ('best_teacher_checkpoint.pth', 'teacher')]:
        old_path = os.path.join(save_path, old_name)
        if os.path.exists(old_path):
            # 避免重复添加（如果已被目录扫描覆盖）
            if not any(p == old_path for _, p, _ in checkpoints):
                checkpoints.append((old_name, old_path, mtype))

    return checkpoints


def _get_state_dict(model):
    """兼容 DataParallel 和普通模型，获取 state_dict。"""
    return model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()


def _save_topk_checkpoint(prefix, state, current_iou, epoch, checkpoint_dir, logger, keep=10):
    """
    保存 Top-K 检查点，自动淘汰 IoU 最低的旧检查点。

    Args:
        prefix (str): 文件名前缀，如 'best_student' 或 'best_teacher'。
        state (dict): 要保存的状态字典。
        current_iou (float): 当前 IoU 值。
        epoch (int): 当前轮数。
        checkpoint_dir (str): 检查点保存目录。
        logger: 日志记录器。
        keep (int): 最多保留的检查点数量，默认 10。

    Returns:
        tuple: (是否保存成功, 保存路径, 被移除的路径列表)
    """
    current_iou = float(current_iou)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 扫描目录中已有的同前缀检查点
    pattern = re.compile(rf'^{re.escape(prefix)}_epoch_(\d+)_iou_([0-9]+(?:\.[0-9]+)?)\.pth$')
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match is None:
            continue
        checkpoints.append({
            'path': os.path.join(checkpoint_dir, filename),
            'epoch': int(match.group(1)),
            'iou': float(match.group(2)),
        })

    # 如果已满且当前 IoU 低于所有已有检查点，跳过保存
    if len(checkpoints) >= keep:
        worst_iou = min(item['iou'] for item in checkpoints)
        if current_iou < worst_iou:
            return False, None, []

    # 保存新检查点
    checkpoint_name = f'{prefix}_epoch_{epoch:03d}_iou_{current_iou:.4f}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(state, checkpoint_path)

    checkpoints.append({
        'path': checkpoint_path,
        'epoch': int(epoch),
        'iou': current_iou,
    })

    # 去重后按 (iou, epoch) 降序排列，淘汰超出 keep 数量的
    unique_checkpoints = {item['path']: item for item in checkpoints}.values()
    ranked_checkpoints = sorted(
        unique_checkpoints,
        key=lambda item: (item['iou'], item['epoch']),
        reverse=True
    )

    removed_paths = []
    for stale in ranked_checkpoints[keep:]:
        try:
            os.remove(stale['path'])
            removed_paths.append(stale['path'])
        except OSError as exc:
            logger.warning(f"Failed to remove stale top-{keep} checkpoint: {stale['path']}, {exc}")

    return os.path.exists(checkpoint_path), checkpoint_path, removed_paths


def save_checkpoint(epoch, model, ema_model, optimizer, lr_scheduler,
                    current_student_iou, current_teacher_iou,
                    save_path, args, logger, use_aspr=False, hooks_dict=None):
    """
    统一的检查点保存方法，同时处理"最新"检查点和"最佳"Top-K检查点。

    Args:
        epoch (int): 当前轮数。
        model (torch.nn.Module): 学生模型。
        ema_model (torch.nn.Module): 教师模型 (EMA)。
        optimizer (torch.optim.Optimizer): 优化器。
        lr_scheduler: 学习率调度器。
        current_student_iou (float): 当前学生模型的验证集 IoU。
        current_teacher_iou (float): 当前教师模型的验证集 IoU。
        save_path (str): 检查点保存目录。
        args: 包含 best_student_iou, best_student_epoch, best_teacher_iou, best_teacher_epoch 的参数对象。
        logger: 日志记录器。
        use_aspr (bool): 是否使用 ASPR（自适应统计伪标签细化）。
        hooks_dict (OrderedDict, optional): hooks 字典，用于获取 ASPR 状态。
    """
    # -----------------------------------------------------------------
    # 1. 检查并更新历史最佳指标
    # -----------------------------------------------------------------
    is_best_student = current_student_iou >= args.best_student_iou
    if is_best_student:
        args.best_student_iou = current_student_iou
        args.best_student_epoch = epoch

    is_best_teacher = current_teacher_iou >= args.best_teacher_iou
    if is_best_teacher:
        args.best_teacher_iou = current_teacher_iou
        args.best_teacher_epoch = epoch

    # -----------------------------------------------------------------
    # 2. 准备"最新"状态字典
    # -----------------------------------------------------------------
    latest_state = {
        'epoch': epoch,
        'pseudo_strategy': getattr(args, 'pseudo_strategy', 'aspr'),
        'model_state_dict': _get_state_dict(model),
        'ema_model_state_dict': _get_state_dict(ema_model),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'best_student_iou': args.best_student_iou,
        'best_student_epoch': args.best_student_epoch,
        'best_teacher_iou': args.best_teacher_iou,
        'best_teacher_epoch': args.best_teacher_epoch,
    }
    if use_aspr and hooks_dict is not None and 'MaskingHook' in hooks_dict:
        masking_hook = hooks_dict['MaskingHook']
        if getattr(args, 'pseudo_strategy', 'aspr') == 'aspr':
            latest_state['prob_max_mu'] = masking_hook.prob_max_mu
            latest_state['prob_max_var'] = masking_hook.prob_max_var
        else:
            latest_state['pseudo_strategy_state'] = masking_hook.state_dict()

    # -----------------------------------------------------------------
    # 3. 保存"最新"检查点
    # -----------------------------------------------------------------
    latest_checkpoint_path = os.path.join(save_path, 'latest_checkpoint.pth')
    torch.save(latest_state, latest_checkpoint_path)
    logger.info(f"轮数 {epoch}: 已保存最新的统一检查点到 {latest_checkpoint_path}")

    # -----------------------------------------------------------------
    # 4. 保存"最佳"Top-K 检查点
    # -----------------------------------------------------------------
    # 学生模型
    best_student_state = {
        'epoch': epoch,
        'val_iou': float(current_student_iou),
        'model_state_dict': _get_state_dict(model),
    }
    student_dir = os.path.join(save_path, 'best_student_checkpoints')
    saved, path, removed = _save_topk_checkpoint(
        'best_student', best_student_state, current_student_iou, epoch, student_dir, logger
    )
    if saved:
        logger.info(f"Epoch {epoch}: student checkpoint entered top-10, IoU: {float(current_student_iou):.4f}. Saved to {path}")
        for r in removed:
            logger.info(f"Removed student checkpoint outside top-10: {r}")

    # 教师模型
    best_teacher_state = {
        'epoch': epoch,
        'val_iou': float(current_teacher_iou),
        'ema_model_state_dict': _get_state_dict(ema_model),
    }
    teacher_dir = os.path.join(save_path, 'best_teacher_checkpoints')
    saved, path, removed = _save_topk_checkpoint(
        'best_teacher', best_teacher_state, current_teacher_iou, epoch, teacher_dir, logger
    )
    if saved:
        logger.info(f"Epoch {epoch}: teacher checkpoint entered top-10, IoU: {float(current_teacher_iou):.4f}. Saved to {path}")
        for r in removed:
            logger.info(f"Removed teacher checkpoint outside top-10: {r}")
