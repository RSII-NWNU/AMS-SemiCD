import os
import argparse
from typing import Dict
# import mmcv
import torch
from ptflops import get_model_complexity_info
# from torch.distributed.checkpoint import state_dict  # 删除这行错误的导入
from tqdm import tqdm
from utils import Evaluator
# from version import __version__
import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils import loggering
from model import SemiModel
from utils import data_loader_new
from utils.save_checkpoint import load_state_dict_from_checkpoint
from PIL import Image
from collections import defaultdict  # 添加这一行

class FeatureExtractor:
    def __init__(self, model, target_layers=None, store_on_cpu=True):
        self.model = model
        self.target_layers = target_layers or []
        self.store_on_cpu = store_on_cpu
        self.features = defaultdict(list)
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        if not self.target_layers:
            return
        module_dict = dict(self.model.named_modules())
        for layer_name in self.target_layers:
            module = module_dict.get(layer_name)
            if module is None:
                print(f"[FeatureExtractor] Warning: layer '{layer_name}' not found.")
                continue
            handle = module.register_forward_hook(self._make_hook(layer_name))
            self.handles.append(handle)

    def _make_hook(self, layer_name):
        def hook(_, __, output):
            feat = output.detach()
            if self.store_on_cpu:
                feat = feat.cpu()
            self.features[layer_name] = [feat]
        return hook

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear_features(self):
        self.features = defaultdict(list)

    def eval(self):
        self.model.eval()

    def train(self, mode=True):
        self.model.train(mode)

class ChangeDetectionVisualizer:
    def __init__(self, dataset_name):
        # 根据数据集名称设置调色板
        if dataset_name == 'LEVIR':
            self.palette = [[0, 0, 0], [255, 255, 255]]  # LEVIR-CD调色板：黑色(无变化)，白色(有变化)
        elif dataset_name == 'WHU':
            self.palette = [[0, 0, 0], [255, 255, 255]]  # WHU-CD调色板：黑色(无变化)，红色(有变化)
        elif dataset_name == 'DSIFN':
            self.palette = [[0, 0, 0], [255, 255, 255]]  # DSIFN-CD调色板
        else:
            # 默认调色板
            self.palette = [[0, 0, 0], [255, 255, 255]]

    def colorize_label(self, seg, palette):
        """为分割结果着色"""
        color_seg = 255 * np.ones((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            if not np.all(color == [255, 255, 255]):
                color_seg[seg == label, :] = color
        return color_seg

    def create_confusion_map(self, gt, pred):
        """创建混淆矩阵可视化图
        TP (True Positive): white [255, 255, 255]
        TN (True Negative): black [0, 0, 0]
        FP (False Positive): red [255, 0, 0]
        FN (False Negative): blue [0, 0, 255]
        """
        # 转换为numpy数组
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        # 确保数组是2D的
        if gt.ndim == 3:
            gt = gt.squeeze()
        if pred.ndim == 3:
            pred = pred.squeeze()

        # 现在可以安全地获取形状
        h, w = gt.shape
        confusion_map = np.zeros((h, w, 3), dtype=np.uint8)

        # 计算各种情况
        tp_mask = (gt == 1) & (pred == 1)  # 真正例
        tn_mask = (gt == 0) & (pred == 0)  # 真负例
        fp_mask = (gt == 0) & (pred == 1)  # 假正例
        fn_mask = (gt == 1) & (pred == 0)  # 假负例

        # 着色
        confusion_map[tp_mask] = [255, 255, 255]  # 白色 - TP
        confusion_map[tn_mask] = [0, 0, 0]  # 黑色 - TN
        confusion_map[fp_mask] = [255, 0, 0]  # 红色 - FP
        confusion_map[fn_mask] = [0, 0, 255]  # 蓝色 - FN

        return confusion_map

    def calculate_metrics(self, gt, pred):
        """计算TP, TN, FP, FN的像素数量"""
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        tp = np.sum((gt == 1) & (pred == 1))
        tn = np.sum((gt == 0) & (pred == 0))
        fp = np.sum((gt == 0) & (pred == 1))
        fn = np.sum((gt == 1) & (pred == 0))

        return tp, tn, fp, fn

    def plot_data(self, ax, title, type, data, palette=None):
        """绘制数据到指定的axes (用于matplotlib)"""
        data = data.cpu() if isinstance(data, torch.Tensor) else data

        if type == 'image':
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            if isinstance(data, torch.Tensor):
                data = data.permute([1, 2, 0]).mul(std).add(mean)
                data = torch.clamp(data, 0, 1)  # 确保值在[0,1]范围内
            ax.imshow(data)
        elif type == 'label':
            if isinstance(data, torch.Tensor):
                data = data.squeeze() if data.dim() > 2 else data
                data = data.numpy()
            out = self.colorize_label(data, palette)
            ax.imshow(out)
        elif type == 'prediction':
            if isinstance(data, torch.Tensor):
                data = data.squeeze(0).argmax(dim=0) if data.dim() > 2 else data
                data = data.numpy()
            out = self.colorize_label(data, palette)
            ax.imshow(out)
        elif type == 'heatmap':
            if isinstance(data, torch.Tensor):
                data = data.squeeze() if data.dim() > 2 else data
                data = data.numpy()
            ax.imshow(data, cmap='gray')
        elif type == 'confusion':
            # data应该已经是RGB图像
            ax.imshow(data)

        if title is not None:
            ax.set_title(title, fontsize=12, pad=10)
        ax.axis('off')

    def save_single_prediction(self, out_data, id, save_dir, filename_suffix=''):
        """单独保存预测结果图片（纯图像，无标签）"""
        os.makedirs(save_dir, exist_ok=True)

        for b_i in range(out_data.shape[0]):
            pred_data = out_data[b_i]
            # 确保数据是2D的
            if len(pred_data.shape) == 3 and pred_data.shape[0] == 1:
                pred_data = pred_data.squeeze(0)
            # 转换为numpy数组并着色
            if isinstance(pred_data, torch.Tensor):
                pred_data = pred_data.cpu().numpy()
            colored_pred = self.colorize_label(pred_data, self.palette)
            # 保存图片（使用cv2直接保存，避免matplotlib的边框和标签）
            base_name = id[b_i].replace('.png', '').replace('.jpg', '')
            save_path = os.path.join(save_dir, f'{base_name}_prediction{filename_suffix}.png')
            # 将RGB转换为BGR（cv2使用BGR格式）
            colored_pred_bgr = cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, colored_pred_bgr)

    def save_single_confusion(self, mask_data, out_data, id, save_dir, filename_suffix=''):
        """单独保存混淆矩阵图片（纯图像，无标签）"""
        os.makedirs(save_dir, exist_ok=True)

        for b_i in range(mask_data.shape[0]):
            gt_data = mask_data[b_i]
            pred_data = out_data[b_i]
            # 确保数据是2D的
            if len(gt_data.shape) == 3 and gt_data.shape[0] == 1:
                gt_data = gt_data.squeeze(0)
            if len(pred_data.shape) == 3 and pred_data.shape[0] == 1:
                pred_data = pred_data.squeeze(0)
            # 创建混淆矩阵可视化
            confusion_map = self.create_confusion_map(gt_data, pred_data)
            # 保存图片（使用cv2直接保存，避免matplotlib的边框和标签）
            base_name = id[b_i].replace('.png', '').replace('.jpg', '')
            save_path = os.path.join(save_dir, f'{base_name}_confusion{filename_suffix}.png')
            # 将RGB转换为BGR（cv2使用BGR格式）
            confusion_map_bgr = cv2.cvtColor(confusion_map, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, confusion_map_bgr)

    def create_heatmap_on_mask(self, prob_map_tensor, mask_tensor, colormap=cv2.COLORMAP_JET, alpha=1):
        """
        将预测概率热图叠加到真实标签（mask）上。

        Args:
            prob_map_tensor (torch.Tensor): 模型的概率输出图，形状为 [H, W]，值在 [0, 1]。
            mask_tensor (torch.Tensor): 真实标签图，形状为 [H, W]，值为 0 或 1。
            colormap (int): OpenCV 的颜色映射。
            alpha (float): 热图的透明度。

        Returns:
            PIL.Image.Image: 叠加后的图像。
        """
        # 1. 准备背景（真实标签）
        # 将 mask 转换为三通道灰度图，变化区域为白色，未变化为黑色
        mask_np = mask_tensor.cpu().numpy()
        background = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
        background[mask_np == 1] = [255, 255, 255]  # 变化区域为白色

        # 2. 准备热图
        prob_map_np = prob_map_tensor.cpu().numpy()
        heatmap_gray = (prob_map_np * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_gray, colormap)
        # OpenCV 使用 BGR，转为 RGB
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # 3. 叠加图像
        overlayed_image = cv2.addWeighted(background, 1 - alpha, heatmap_color, alpha, 0)

        return Image.fromarray(overlayed_image)

    def save_heatmap_on_masks(self, prob_maps, masks, ids, save_dir, suffix=''):
        """批量保存叠加了热图的mask图像"""
        os.makedirs(save_dir, exist_ok=True)

        prob_maps = prob_maps.cpu()
        masks = masks.cpu()

        for i in range(prob_maps.shape[0]):
            # 从文件名中提取基本名称
            base_name = ids[i].replace('.png', '').replace('.jpg', '')
            # 确保张量是二维的
            prob_map_single = prob_maps[i].squeeze()
            mask_single = masks[i].squeeze()

            # create_heatmap_on_mask 返回的是 RGB 格式的 PIL.Image
            overlayed_pil = self.create_heatmap_on_mask(prob_map_single, mask_single)

            # --- 修正部分 ---
            # 1. 将 PIL Image (RGB) 转换为 NumPy 数组
            overlayed_np_rgb = np.array(overlayed_pil)
            # 2. 将 RGB 转换为 BGR，因为 cv2.imwrite 按 BGR 格式保存
            overlayed_np_bgr = cv2.cvtColor(overlayed_np_rgb, cv2.COLOR_RGB2BGR)

            # 3. 使用 cv2.imwrite 保存
            save_path = os.path.join(save_dir, f"{base_name}{suffix}.png")
            cv2.imwrite(save_path, overlayed_np_bgr)

    def save_feature_maps(self, features: Dict[str, torch.Tensor], ids: list, save_dir: str):
        """
        将提取的特征图作为热力图保存到磁盘 (使用OpenCV以提高效率)。

        Args:
            features (Dict[str, torch.Tensor]): 从钩子中提取的特征图字典。
            ids (list): 批次中每个样本的ID。
            save_dir (str): 保存特征图的目录。
        """
        # 确保主特征图目录存在
        os.makedirs(save_dir, exist_ok=True)

        batch_size = len(ids)
        for b_i in range(batch_size):
            # 获取不带扩展名的基本文件名
            base_name = ids[b_i].replace('.png', '').replace('.jpg', '')

            for name, feat_tensor in features.items():
                # 提取当前样本的特征图 [1, C, H, W]
                sample_feat = feat_tensor[b_i:b_i + 1]
                # 计算均值热力图 [H, W]
                heatmap = sample_feat.detach().cpu().squeeze(0).mean(dim=0).numpy()

                # 归一化到 0-255
                heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # 应用色彩映射
                heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_VIRIDIS)

                # 构建新的文件名并保存图像
                save_path = os.path.join(save_dir, f'{base_name}_{name}.png')
                cv2.imwrite(save_path, heatmap_colored)

    def generate_visualization(self, imgA_x, imgB_x, mask_x, out, id, save_dir):
        """
        生成详细的可视化图片。
        注意：此方法使用matplotlib，性能较低。如果需要批量生成，建议使用或实现基于OpenCV的优化版本。
        """
        os.makedirs(save_dir, exist_ok=True)

        for b_i in range(imgA_x.shape[0]):
            rows, cols = 2, 2  # 2x2布局

            # 数据准备
            imgA_data = imgA_x[b_i]  # [3, 256, 256]
            imgB_data = imgB_x[b_i]  # [3, 256, 256]
            mask_data = mask_x[b_i]  # [256, 256]
            out_data = out[b_i]  # [256, 256]

            # 确保mask是2D的
            if len(mask_data.shape) == 3 and mask_data.shape[0] == 1:
                mask_data = mask_data.squeeze(0)
            if len(out_data.shape) == 3 and out_data.shape[0] == 1:
                out_data = out_data.squeeze(0)

            # 创建混淆矩阵可视化
            confusion_map = self.create_confusion_map(mask_data, out_data)

            # 计算指标
            tp, tn, fp, fn = self.calculate_metrics(mask_data, out_data)

            # 创建包含指标信息的标题
            confusion_title = f'Confusion Map\nTP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}'

            plot_dicts = [
                dict(title='Image A', data=imgA_data, type='image'),
                dict(title='Image B', data=imgB_data, type='image'),
                dict(title='Ground Truth', data=mask_data, type='label', palette=self.palette),
                dict(title='Prediction', data=out_data, type='label', palette=self.palette),
            ]

            # 创建图形
            fig, axs = plt.subplots(
                rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False,
                gridspec_kw={'hspace': 0.3, 'wspace': 0.2, 'top': 0.85, 'bottom': 0.15, 'right': 0.9, 'left': 0.1}
            )

            # 绘制前4个子图
            for i, (ax, plot_dict) in enumerate(zip(axs.flat, plot_dicts)):
                self.plot_data(ax, **plot_dict)

            # 在整个图的底部添加混淆矩阵图
            # 移除最后一行的子图，创建一个跨列的子图
            axs[1, 0].remove()
            axs[1, 1].remove()

            # 创建跨列的混淆矩阵子图
            confusion_ax = fig.add_subplot(2, 1, 2)
            confusion_ax.imshow(confusion_map)
            confusion_ax.set_title(confusion_title, fontsize=14, pad=20)
            confusion_ax.axis('off')

            # 添加图例
            legend_text = 'TP: White | TN: Black | FP: Red | FN: Blue'
            confusion_ax.text(0.5, -0.1, legend_text, transform=confusion_ax.transAxes,
                              ha='center', va='top', fontsize=12,
                              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            # 保存图片
            save_path = os.path.join(save_dir, f'{id[b_i]}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

    def _create_visualization_grid(self, img_a_vis, img_b_vis, mask_data, out_data, id_single):
        """使用OpenCV为单个样本创建2x3的可视化网格。"""
        # OpenCV字体和颜色设置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color_title = (255, 255, 255)  # 标题用白色
        font_color_metrics = (0, 0, 0)  # 指标用黑色
        line_type = 1
        title_bg_color = (0, 0, 0)  # 黑色背景
        title_bar_height = 40

        # --- 1. 准备单个图像 ---
        img_a = cv2.cvtColor(img_a_vis, cv2.COLOR_RGB2BGR)
        img_b = cv2.cvtColor(img_b_vis, cv2.COLOR_RGB2BGR)
        gt_np = mask_data.squeeze().cpu().numpy()
        gt_colored = self.colorize_label(gt_np, self.palette)
        gt_colored = cv2.cvtColor(gt_colored, cv2.COLOR_RGB2BGR)
        pred_np = out_data.squeeze().cpu().numpy()
        pred_colored = self.colorize_label(pred_np, self.palette)
        pred_colored = cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR)
        confusion_map = self.create_confusion_map(mask_data, out_data)
        confusion_map = cv2.cvtColor(confusion_map, cv2.COLOR_RGB2BGR)

        # --- 2. 添加标题 ---
        def add_title(image, title):
            new_img = cv2.copyMakeBorder(image, title_bar_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=title_bg_color)
            text_size = cv2.getTextSize(title, font, font_scale, line_type)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = (title_bar_height + text_size[1]) // 2
            cv2.putText(new_img, title, (text_x, text_y), font, font_scale, font_color_title, line_type)
            return new_img

        img_a_titled = add_title(img_a, 'Image A')
        img_b_titled = add_title(img_b, 'Image B')
        gt_titled = add_title(gt_colored, 'Ground Truth')
        pred_titled = add_title(pred_colored, 'Prediction')
        confusion_titled = add_title(confusion_map, 'Confusion Map')

        # --- 3. 创建占位符并填入指标 ---
        h, w, _ = img_a_titled.shape
        placeholder = np.full((h, w, 3), 255, dtype=np.uint8)  # 白色占位符

        tp, tn, fp, fn = self.calculate_metrics(mask_data, out_data)
        metrics_text = [f"TP: {tp}", f"TN: {tn}", f"FP: {fp}", f"FN: {fn}"]

        y_offset = h // 2 - 40
        for i, line in enumerate(metrics_text):
            text_size = cv2.getTextSize(line, font, font_scale, line_type)[0]
            text_x = (w - text_size[0]) // 2
            text_y = y_offset + i * 25
            cv2.putText(placeholder, line, (text_x, text_y), font, font_scale, font_color_metrics, line_type)

        # --- 4. 拼接图像 ---
        row1 = np.hstack((img_a_titled, img_b_titled, gt_titled))
        row2 = np.hstack((pred_titled, confusion_titled, placeholder))
        grid = np.vstack((row1, row2))

        # --- 5. 添加底部图例 ---
        legend_text = 'TP: White | TN: Black | FP: Red | FN: Blue'
        legend_bar_height = 30
        grid = cv2.copyMakeBorder(grid, 0, legend_bar_height, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        grid_h, grid_w, _ = grid.shape
        text_size = cv2.getTextSize(legend_text, font, 0.6, line_type)[0]
        text_x = (grid_w - text_size[0]) // 2
        text_y = grid_h - (legend_bar_height - text_size[1]) // 2
        cv2.putText(grid, legend_text, (text_x, text_y), font, 0.6, (0, 0, 0), line_type)

        return grid

    def generate_simple_visualization(self, imgA_x, imgB_x, mask_x, out, id, save_dir):
        """生成简化版可视化图片并保存 (使用OpenCV优化以提高速度)"""
        os.makedirs(save_dir, exist_ok=True)

        # 批量反归一化 (在CPU上操作)
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgA_x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgA_x.device).view(1, 3, 1, 1)
        imgA_vis = (imgA_x.mul(std).add(mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        imgB_vis = (imgB_x.mul(std).add(mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

        for b_i in range(imgA_x.shape[0]):
            grid = self._create_visualization_grid(
                imgA_vis[b_i], imgB_vis[b_i], mask_x[b_i], out[b_i], id[b_i]
            )
            base_name = id[b_i].replace('.png', '').replace('.jpg', '')
            save_path = os.path.join(save_dir, f'{base_name}.png')
            cv2.imwrite(save_path, grid)

    def display_batch_grid(self, imgA_x, imgB_x, mask_x, out, id):
        """使用Matplotlib生成并实时显示整个批次的可视化网格"""
        all_grids = []
        # 批量反归一化
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgA_x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgA_x.device).view(1, 3, 1, 1)
        imgA_vis = (imgA_x.mul(std).add(mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        imgB_vis = (imgB_x.mul(std).add(mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

        for b_i in range(imgA_x.shape[0]):
            grid = self._create_visualization_grid(
                imgA_vis[b_i], imgB_vis[b_i], mask_x[b_i], out[b_i], id[b_i]
            )
            all_grids.append(grid)

        if not all_grids:
            print("没有可供显示的图像。")
            return

        final_image = np.vstack(all_grids)
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

        dpi = 100
        fig_h = final_image_rgb.shape[0] / dpi
        fig_w = final_image_rgb.shape[1] / dpi
        plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        plt.imshow(final_image_rgb)
        plt.axis('off')
        plt.title(f'Random Batch Visualization (Batch Size: {imgA_x.shape[0]})', fontsize=16)
        plt.tight_layout()
        plt.show()

    def display_batch_heatmaps(self, prob_maps, masks, ids):
        """
        为整个批次创建并实时显示热图叠加的可视化结果。
        """
        all_heatmaps = []
        prob_maps = prob_maps.cpu()
        masks = masks.cpu()

        for i in range(prob_maps.shape[0]):
            prob_map_single = prob_maps[i].squeeze()
            mask_single = masks[i].squeeze()

            # 创建叠加后的图像 (返回 PIL.Image)
            overlayed_pil = self.create_heatmap_on_mask(prob_map_single, mask_single)

            # 将 PIL 图像转换为 NumPy 数组以便使用 OpenCV 添加标题
            overlayed_np = np.array(overlayed_pil)
            overlayed_bgr = cv2.cvtColor(overlayed_np, cv2.COLOR_RGB2BGR)

            # 添加标题栏
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            title_bar_height = 30
            titled_img = cv2.copyMakeBorder(overlayed_bgr, title_bar_height, 0, 0, 0, cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0))
            title = f"Heatmap on GT: {ids[i].replace('.png', '')}"
            text_size = cv2.getTextSize(title, font, font_scale, 1)[0]
            text_x = (overlayed_bgr.shape[1] - text_size[0]) // 2
            text_y = (title_bar_height + text_size[1]) // 2
            cv2.putText(titled_img, title, (text_x, text_y), font, font_scale, (255, 255, 255), 1)

            all_heatmaps.append(titled_img)

        if not all_heatmaps:
            print("没有可供显示的热图。")
            return

        # 垂直拼接所有图像
        final_image = np.vstack(all_heatmaps)
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

        # 使用 Matplotlib 显示
        dpi = 100
        fig_h = final_image_rgb.shape[0] / dpi
        fig_w = final_image_rgb.shape[1] / dpi
        plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        plt.imshow(final_image_rgb)
        plt.axis('off')
        plt.suptitle(f'Heatmap on Mask (Batch Size: {prob_maps.shape[0]})', fontsize=14)
        plt.tight_layout()
        plt.show()

    def display_batch_with_features(self, imgA_x, imgB_x, gt_x, pred_x, features: Dict[str, torch.Tensor], ids: list):
        """
        为整个批次创建并显示包含特征图的可视化结果。
        """
        batch_size = imgA_x.shape[0]
        num_features = len(features)
        num_base_images = 5  # A, B, GT, Pred, Confusion
        cols = num_base_images + num_features

        fig, axs = plt.subplots(batch_size, cols, figsize=(cols * 3, batch_size * 3), squeeze=False)
        fig.suptitle('Feature Visualization for Batch', fontsize=24)

        # 反归一化参数
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        for b_i in range(batch_size):
            # --- 1. 准备基础图像 ---
            imgA_vis = (imgA_x[b_i:b_i + 1].cpu().mul(std).add(mean).clamp(0, 1).permute(0, 2, 3, 1).squeeze(
                0).numpy() * 255).astype(np.uint8)
            imgB_vis = (imgB_x[b_i:b_i + 1].cpu().mul(std).add(mean).clamp(0, 1).permute(0, 2, 3, 1).squeeze(
                0).numpy() * 255).astype(np.uint8)
            gt_vis = self.colorize_label(gt_x[b_i].cpu().squeeze().numpy(), self.palette)
            pred_vis = self.colorize_label(pred_x[b_i].cpu().squeeze().numpy(), self.palette)
            confusion_vis = self.create_confusion_map(gt_x[b_i], pred_x[b_i])

            base_images = {'Image A': imgA_vis, 'Image B': imgB_vis, 'GT': gt_vis, 'Pred': pred_vis,
                           'Confusion': confusion_vis}

            # --- 2. 绘制基础图像 ---
            for i, (title, img) in enumerate(base_images.items()):
                ax = axs[b_i, i]
                ax.imshow(img)
                ax.axis('off')
                if b_i == 0:  # 仅在第一行显示标题
                    ax.set_title(title, fontsize=10)
            axs[b_i, 0].text(-0.1, 0.5, ids[b_i], transform=axs[b_i, 0].transAxes, ha="right", va="center", fontsize=10,
                             rotation=90)

            # --- 3. 绘制特征图 ---
            for i, (name, feat_tensor) in enumerate(features.items()):
                ax = axs[b_i, num_base_images + i]
                heatmap = feat_tensor[b_i].detach().cpu().mean(dim=0).numpy()
                ax.imshow(heatmap, cmap='viridis')
                ax.axis('off')
                if b_i == 0:  # 仅在第一行显示标题
                    ax.set_title(name, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()


def custom_input_constructor(input_shape) -> Dict[str, torch.Tensor]:
    # 示例：生成两个相同形状的随机张量
    batch_size = 1  # 默认为1以简化计算
    A = torch.randn(batch_size, *input_shape, device=device)
    B = torch.randn(batch_size, *input_shape, device=device)
    return {
        'A': A,
        'B': B
    }


def test(args, test_loader, evaluator, visualizer):
    # 初始化模型
    model = args.model
    device = next(model.parameters()).device
    model.eval()
    # 定义输入形状（例如：3通道、256x256图像）
    input_shape = (3, 256, 256)

    # 计算 FLOPs 和参数量
    macs, params = get_model_complexity_info(
        model,
        input_res=input_shape,  # 输入形状（需与构造函数中的形状匹配）
        input_constructor=custom_input_constructor,  # 指定自定义输入
        as_strings=True,  # 结果格式化为字符串
        print_per_layer_stat=False  # 关闭逐层统计（可选）
    )

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
    checkpoint_path = os.path.join(args.save_path, args.path_name)
    ckpt_type = getattr(args, 'checkpoint_type', 'auto')
    state_dict = load_state_dict_from_checkpoint(checkpoint_path, ckpt_type)
            
    # 重新加载权重到模型
    model.load_state_dict(state_dict)
    model.eval() # 再次确保是 eval 模式
    # ========================================================

    # 初始化时间统计变量
    total_inference_time = 0.0  # 累计纯推理时间
    total_samples = 0           # 累计样本数量

    # 定义要使用钩子提取特征的目标层
    # target_layers = ['dbdfe1', 'sam_p1', 'conv4', 'out_conv']
    target_layers = []
    feature_extractor = FeatureExtractor(model, target_layers)
    feature_extractor.eval()

    # 计数器
    vis_count = 0
    separate_count = 0

    with torch.no_grad():
        for batch_idx, (imgA_x, imgB_x, mask_x, id) in enumerate(tqdm(test_loader, desc="Testing")):
            imgA_x, imgB_x, mask_x = imgA_x.to(device), imgB_x.to(device), mask_x.to(device)
            total_samples += imgA_x.size(0)

            # ------------------- 新增：精确计时逻辑 -------------------
            if torch.cuda.is_available():
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()  # 记录开始
                
                # 模型推理，获取直接返回的特征和输出
                model_output = model(imgA_x, imgB_x)
                
                ender.record()    # 记录结束
                torch.cuda.synchronize()  # 等待 GPU 完成所有操作
                curr_time = starter.elapsed_time(ender) / 1000.0  # 转换为秒
            else:
                start_time = time.time()
                # 模型推理，获取直接返回的特征和输出
                model_output = model(imgA_x, imgB_x)
                end_time = time.time()
                curr_time = end_time - start_time
            
            total_inference_time += curr_time
            # ---------------------------------------------------------

            # --- 解析模型输出 ---
            intermediate_features = {}
            if isinstance(model_output, tuple) and len(model_output) > 0:
                # 假定最后一个张量是主输出
                out = model_output[-1] if not isinstance(model_output[-1], dict) else model_output[-2]
                # 假定最后一个元素可能是特征字典
                if isinstance(model_output[-1], dict):
                    intermediate_features = model_output[-1]
            else:
                out = model_output

            if not isinstance(out, torch.Tensor):
                raise ValueError(f"Model output could not be resolved to a tensor, got {type(out)}")

            # --- 获取概率图和二值图 ---
            if out.shape[1] == 1:
                out_prob = torch.sigmoid(out)
                out_argmax = (out_prob > 0.5).int()
            else:
                out_prob = torch.softmax(out, dim=1)[:, 1, :, :].unsqueeze(1)
                out_argmax = torch.argmax(out, dim=1)

            # --- 合并所有特征图 ---
            processed_features = intermediate_features.copy()
            hooked_features = feature_extractor.features
            for name, feat_list in hooked_features.items():
                if feat_list:
                    processed_features[name] = feat_list[0]

            # --- 实时可视化第一个批次 ---
            if args.display_first_batch and batch_idx == 0:
                print("\n正在显示第一个批次的可视化结果。关闭图像窗口后测试将继续...")
                visualizer.display_batch_grid(imgA_x.cpu(), imgB_x.cpu(), mask_x.cpu(), out_argmax.cpu(), id)
                visualizer.display_batch_heatmaps(out_prob.cpu(), mask_x.cpu(), id)
                if processed_features:
                    visualizer.display_batch_with_features(imgA_x, imgB_x, mask_x, out_argmax, processed_features, id)

            # 计算指标
            evaluator.add_batch(mask_x, out_argmax)

            # --- 保存综合可视化图片 ---
            if args.save_vis and (args.vis_batches == -1 or batch_idx < args.vis_batches):
                vis_save_dir = os.path.join(args.save_dir, 'visualizations')
                if args.vis_layout == 'detailed':
                    visualizer.generate_visualization(imgA_x.cpu(), imgB_x.cpu(), mask_x.cpu(), out_argmax.cpu(), id,
                                                      vis_save_dir)
                else:
                    visualizer.generate_simple_visualization(imgA_x.cpu(), imgB_x.cpu(), mask_x.cpu(), out_argmax.cpu(),
                                                             id, vis_save_dir)
                vis_count += imgA_x.shape[0]

            # --- 根据目录参数独立保存各类图片 ---
            if args.save_separate and (args.vis_batches == -1 or batch_idx < args.vis_batches):
                has_saved = False
                # 保存预测图
                if args.pred_dir:
                    pred_save_dir = os.path.join(args.save_dir, args.pred_dir)
                    visualizer.save_single_prediction(out_argmax.cpu(), id, pred_save_dir, args.filename_suffix)
                    has_saved = True
                # 保存混淆矩阵
                if args.confusion_dir:
                    confusion_save_dir = os.path.join(args.save_dir, args.confusion_dir)
                    visualizer.save_single_confusion(mask_x.cpu(), out_argmax.cpu(), id, confusion_save_dir,
                                                     args.filename_suffix)
                    has_saved = True
                # 保存热图
                if args.heatmap_dir:
                    heatmap_save_dir = os.path.join(args.save_dir, args.heatmap_dir)
                    visualizer.save_heatmap_on_masks(out_prob, mask_x, id, heatmap_save_dir, args.filename_suffix)
                    has_saved = True
                # 保存特征图
                if args.feature_dir and processed_features:
                    feature_save_dir = os.path.join(args.save_dir, args.feature_dir)
                    visualizer.save_feature_maps(processed_features, id, feature_save_dir)
                    has_saved = True

                if has_saved:
                    separate_count += imgA_x.shape[0]
            # if args.vis_batches <= (vis_count/args.batch_size) or args.vis_batches != -1:
            #      break
    feature_extractor.remove_hooks()

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
    
    IoU = evaluator.Intersection_over_Union()
    Pre = evaluator.Precision()
    Rec = evaluator.Recall()
    F1_score = evaluator.F1()
    FA = evaluator.False_Alarm_Rate()
    MA = evaluator.Missed_Detection_Rate()
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

    if args.save_vis:
        print(f"Visualization images saved to: {os.path.join(args.save_dir, 'visualizations')}")
    if args.save_separate:
        print(f"Separately saved images count: {separate_count}")
        if args.pred_dir: print(f"  - Predictions saved to: {os.path.join(args.save_dir, args.pred_dir)}")
        if args.confusion_dir: print(f"  - Confusion maps saved to: {os.path.join(args.save_dir, args.confusion_dir)}")
        if args.heatmap_dir: print(f"  - Heatmaps saved to: {os.path.join(args.save_dir, args.heatmap_dir)}")
        if args.feature_dir: print(f"  - Feature maps saved to: {os.path.join(args.save_dir, args.feature_dir)}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str,
                        default='./output/C2F-SemiCD/LEVIR/SemiModel_ALL',
                        help='Path to saved models')
    parser.add_argument('--path_name', type=str, default='best_student_model.pth',
                        help='Path to saved models')
    parser.add_argument('--checkpoint_type', type=str, default='auto',
                        choices=['student', 'teacher', 'auto'],
                        help='Checkpoint type: student (model_state_dict), teacher (ema_model_state_dict), auto')
    parser.add_argument('--save_dir', type=str, default='./test_results/LEVIR-NoAGAR', required=False)
    parser.add_argument('--save_vis', type=bool, default=True, help='Save visualization images')
    parser.add_argument('--vis_batches', type=int, default=1, help='要可视化的批次数 (-1 表示全部)')
    parser.add_argument('--vis_layout', type=str, default='simple', choices=['simple', 'detailed'],
                        help='Visualization layout: simple (2x3) or detailed (2x2 with large confusion map)')
    parser.add_argument('--color', type=str, default='LEVIR', required=False)

    # 新增参数：单独保存预测和混淆矩阵
    parser.add_argument('--save_separate', type=bool, default=True,
                        help='Save prediction and confusion maps separately')
    parser.add_argument('--pred_dir', type=str, default=None,
                        help='Directory name for saving prediction images (relative to save_dir)')
    parser.add_argument('--confusion_dir', type=str, default='confusion_maps',
                        help='Directory name for saving confusion map images (relative to save_dir)')
    parser.add_argument('--feature_dir', type=str, default=None,
                        help='Directory name for saving confusion map images (relative to save_dir)')
    parser.add_argument('--heatmap_dir', type=str, default='heatmaps',
                        help='Directory for saving heatmaps on masks. If empty, will not save.')
    parser.add_argument('--filename_suffix', type=str, default='',
                        help='Suffix to add to individual prediction and confusion filenames')

    # 在此处添加新参数
    parser.add_argument('--display_first_batch', type=bool, default=True,
                        help='在测试开始时显示第一个批次的可视化图像。')

    # 模型参数
    parser.add_argument('--model_name', type=str, choices=['SemiModel_ALL'], default='SemiModel_ALL',
                        help='Model name to load')
    parser.add_argument('--dataset_name', type=str, default='LEVIR',
                        help='Dataset name for palette selection')
    parser.add_argument('--test_root', type=str, default='./data/LEVIR-CD-256/test/',
                        help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--train_size', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, default='0')

    args = parser.parse_args()

    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"使用GPU {args.gpu_id} 进行推理")
    else:
        device = torch.device('cpu')
        print("CUDA不可用，使用CPU进行推理")

    # 初始化模型
    if args.model_name != 'SemiModel_ALL':
        raise ValueError("仅支持正式模型 SemiModel_ALL")
    args.model = SemiModel(None, False).to(device)

    # 记录训练日志
    log_path = './log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = os.path.join(log_path, 'test.log')
    logger = loggering(log_path)
    logger.info('PyTorch Version {}\n Experiment{}'.format(torch.__version__, log_path))
    args.logger = logger

    # 加载模型权重
    checkpoint_path = os.path.join(args.save_path, args.path_name)
    model_state_dict = load_state_dict_from_checkpoint(checkpoint_path, args.checkpoint_type)

    # 处理多GPU到单GPU的状态字典转换
    new_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    args.model.load_state_dict(new_state_dict)

    # 加载测试数据集
    test_loader = data_loader_new.get_val_loader(args.test_root, args.batch_size, args.train_size, args.logger,
                                                 num_workers=4,
                                                 shuffle=False, pin_memory=True)

    # 初始化评估器和可视化器
    evaluator = Evaluator(num_class=2)
    visualizer = ChangeDetectionVisualizer(args.dataset_name)

    # 运行测试
    test(args, test_loader, evaluator, visualizer)
