import os
from typing import Dict, Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from ptflops import get_model_complexity_info
import torch
import argparse
from tqdm import tqdm
from utils import *
from model import SemiModel
import sys

# sys.argv = ['msce_test.py']
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_input_constructor(input_shape) -> Dict[str, torch.Tensor]:
    # 示例：生成两个相同形状的随机张量
    batch_size = 1  # 默认为1以简化计算
    A = torch.randn(batch_size, *input_shape, device=device)
    B = torch.randn(batch_size, *input_shape, device=device)
    return {
        'A': A,
        'B': B
    }


# 定义常量
PLOT_PARAMS = {
    'attention': {'figsize': (8, 6), 'dpi': 150},
    'comparison': {'figsize': (10, 5), 'dpi': 150},
    'overlay': {'figsize': (8, 8), 'dpi': 300}
}


def visualize_comparison(pred: np.ndarray, target: np.ndarray,
                         filenames: List[str], save_path: str):
    """可视化预测与标签对比图（优化版）"""
    for j in range(pred.shape[0]):
        try:
            # 生成安全文件名
            safe_name = filenames[j].replace('/', '_').replace('..', '_')

            # 创建画布
            plt.figure(**PLOT_PARAMS['comparison'])

            # 预测图
            plt.subplot(1, 2, 1)
            plt.imshow(pred[j, 0], cmap='gray', vmin=0, vmax=1)
            plt.title('Prediction')
            plt.axis('off')

            # 标签图
            plt.subplot(1, 2, 2)
            plt.imshow(target[j, 0], cmap='gray', vmin=0, vmax=1)
            plt.title('Ground Truth')
            plt.axis('off')

            # 保存对比图
            plt.savefig(os.path.join(save_path, f"{safe_name}_compare.png"),
                        bbox_inches='tight', pad_inches=0)
            plt.close()
        except Exception as e:
            print(f"生成对比图 {filenames[j]} 时出错: {str(e)}")


def model(args):
    # 初始化评估器
    evaluator = Evaluator(num_class=2)

    # 加载测试数据集
    test_loader = get_val_loader(args.test_root, args.batchsize, args.trainsize, args.logger, num_workers=4,
                                             shuffle=False, pin_memory=True)

    # 初始化模型
    model = args.model
    # 定义输入形状（例如：3通道、224x224图像）
    input_shape = (3, 256, 256)

    # 计算 FLOPs 和参数量
    macs, params = get_model_complexity_info(
        model,
        input_res=input_shape,  # 输入形状（需与构造函数中的形状匹配）
        input_constructor=custom_input_constructor,  # 指定自定义输入
        as_strings=True,  # 结果格式化为字符串
        print_per_layer_stat=False  # 关闭逐层统计（可选）
    )

    model.eval()

    # ------------------- 新增：GPU 预热 (Warm-up) -------------------
    # 目的：让 GPU 完成初始化，避免第一次推理耗时过长影响统计
    if torch.cuda.is_available():
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        print("正在进行 GPU 预热...")
        with torch.no_grad():
            for _ in range(50):
                _ = model(dummy_input, dummy_input)
        torch.cuda.synchronize()  # 等待预热完成
    # ---------------------------------------------------------------

    # =========== 【核心修复：预热后重置模型状态】 =============
    # 重新加载权重，清除预热阶段可能产生的内部状态残留
    state_dict = load_state_dict_from_checkpoint(args.path_name, args.checkpoint_type)
            
    # 重新加载权重到模型
    model.load_state_dict(state_dict)
    model.eval() # 再次确保是 eval 模式
    # ========================================================

    # 初始化时间统计变量
    total_inference_time = 0.0  # 累计纯推理时间
    total_samples = 0           # 累计样本数量

    pred_path = os.path.join('./test_result', args.model_name)
    if not os.path.exists(pred_path):
        os.makedirs(pred_path, exist_ok=True)

    # 开始验证
    with torch.no_grad():
        for A, B, mask, filename in tqdm(test_loader, desc="Testing"):
            A = A.to(device)
            B = B.to(device)
            y_true = mask.to(device)
            
            batch_size = A.size(0)
            total_samples += batch_size

            # ------------------- 新增：精确计时逻辑 -------------------
            if torch.cuda.is_available():
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()  # 记录开始
                
                preds = model(A, B)[1]  # 执行推理
                
                ender.record()    # 记录结束
                torch.cuda.synchronize()  # 等待 GPU 完成所有操作
                curr_time = starter.elapsed_time(ender) / 1000.0  # 转换为秒
            else:
                start_time = time.time()
                preds = model(A, B)[1]
                end_time = time.time()
                curr_time = end_time - start_time
            
            total_inference_time += curr_time
            # ---------------------------------------------------------

            outputs = torch.sigmoid(preds)
            preds = (outputs > 0.5).int()

            # 计算指标
            evaluator.add_batch(
                y_true,
                preds
            )

    # ------------------- 新增：计算并打印速度指标 -------------------
    fps = total_samples / total_inference_time
    avg_latency = (total_inference_time / total_samples) * 1000 # 毫秒/张
    
    speed_info = f"""
    === 推理速度统计 ===
    Total Samples: {total_samples}
    Total Time:    {total_inference_time:.4f} s
    FPS:           {fps:.2f} frames/s
    Avg Latency:   {avg_latency:.2f} ms/sample
    """
    print(speed_info)
    # ---------------------------------------------------------------

    # 获取变化类（类别1）的指标
    IoU = evaluator.Intersection_over_Union()
    Pre = evaluator.Precision()
    Rec = evaluator.Recall()
    F1_score = evaluator.F1()
    FA = evaluator.False_Alarm_Rate()
    MA = evaluator.Missed_Detection_Rate()

    # 整体指标
    OA = evaluator.OA()
    Kappa = evaluator.Kappa()

    # 输出表格化结果
    result_log = f"""
    === 变化检测指标（类别1） ===
    IoU:       {IoU[1]:.4f}  # 变化区域的交并比
    Precision: {Pre[1]:.4f}  # 变化类精确率
    Recall:    {Rec[1]:.4f}  # 变化类召回率
    F1 Score:  {F1_score[1]:.4f}  # 变化类F1
    FA Rate:   {FA[1]:.4f}  # 虚警率
    MA Rate:   {MA[1]:.4f}  # 漏检率
    OA:        {OA:.4f}  # 总体分类精度
    Kappa:     {Kappa:.4f}  # 一致性检验
    FLOPs: {macs} | Params: {params}
    FPS: {fps:.2f} | Latency: {avg_latency:.2f} ms
    """
    # print(result_log)
    logger.info(result_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 必须参数
    parser.add_argument('--test_root', type=str, default='/data/proj/ypc/ams-new/data/WHU-CD-256/test/',
                        help='Path to test dataset')
    parser.add_argument('--save_path', type=str, default='/data/proj/ypc/ams-new/output/C2F-SemiCD/WHU/SemiModel_ALL_WHU_0.05_20260102-124615',
                        help='Path to saved models')
    parser.add_argument('--path_name', type=str, default='modified_teacher_checkpoint.pth',
                        help='Path to saved models')
    parser.add_argument('--model_name', type=str, choices=['SemiModel_ALL'], default='SemiModel_ALL',
                        help='Path to saved models')

    # 可选参数（需与训练时一致）
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--trainsize', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, default='0')

    args = parser.parse_args()

    # 强制让程序只能看到你指定的那个 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # 设置设备
    if torch.cuda.is_available():
        # 【关键修改】：因为已经设置了可见性，对于当前进程来说，
        # 这张卡永远是第 0 号卡 (cuda:0)。
        # 不要再使用 f'cuda:{args.gpu_id}'，那会指向不存在的设备索引。
        device = torch.device('cuda:0')
        
        print(f"物理 GPU ID: {args.gpu_id}, 逻辑 GPU ID: 0, 已开启严格隔离。")
    else:
        device = torch.device('cpu')
        print("CUDA不可用，使用CPU进行推理")

    if args.model_name != 'SemiModel_ALL':
        raise ValueError("仅支持正式模型 SemiModel_ALL")
    args.model = SemiModel(None, False).to(device)

    args.save_path = os.path.join(args.save_path)

    # 记录训练日志
    log_path = './log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = os.path.join(log_path, 'train_test.log')
    logger = loggering(log_path)
    logger.info('PyTorch Version {}\n Experiment{}'.format(torch.__version__, log_path))
    args.logger = logger

    # 设置GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # 自动扫描目录下所有检查点（新格式 + 旧格式兼容）
    checkpoints = collect_checkpoints(args.save_path)

    if not checkpoints:
        print(f"在 {args.save_path} 下未找到任何检查点文件，请检查路径。")
    else:
        print(f"共找到 {len(checkpoints)} 个检查点，开始逐一测试...\n")

    for ckpt_name, full_path, ckpt_type in checkpoints:
        print(f"正在测试权重: {ckpt_name} (类型: {ckpt_type})")
        args.path_name = full_path
        args.checkpoint_type = ckpt_type

        # 加载参数到模型
        state_dict = load_state_dict_from_checkpoint(full_path, ckpt_type)
        args.model.load_state_dict(state_dict)

        # 运行模型测试
        model(args)

        # 对于 latest_checkpoint，额外测试 EMA 模型（如果有）
        if ckpt_name == 'latest_checkpoint.pth':
            checkpoint = torch.load(full_path, weights_only=False, map_location='cpu')
            if 'ema_model_state_dict' in checkpoint:
                print(f"正在测试权重: {ckpt_name} (EMA)")
                args.checkpoint_type = 'teacher'
                args.model.load_state_dict(checkpoint['ema_model_state_dict'])
                model(args)

        print()  # 空行分隔不同检查点的输出
