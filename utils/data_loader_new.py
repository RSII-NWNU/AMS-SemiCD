import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import TrivialAugmentWide
import numpy as np
import random
from PIL import ImageEnhance
from torchvision.transforms import functional as F
import logging

# 建议在您的主脚本中配置logging，但在这里获取logger是安全的
# logger = logging.getLogger(__name__)

COLOR_OPS = [
    lambda img, mag: F.autocontrast(img),
    lambda img, mag: F.equalize(img),
    # lambda img, mag: F.invert(img), # Invert通常效果不佳，可以排除
    lambda img, mag: F.posterize(img, bits=max(1, int(4 - mag * 3.99 / 10))),
    lambda img, mag: F.solarize(img, threshold=256 - mag * 25.5),
    lambda img, mag: F.adjust_sharpness(img, sharpness_factor=1 + mag * 0.09),
    lambda img, mag: F.adjust_brightness(img, brightness_factor=1 + mag * 0.09),
    lambda img, mag: F.adjust_contrast(img, contrast_factor=1 + mag * 0.18),
    lambda img, mag: F.adjust_saturation(img, saturation_factor=1 + mag * 0.18),
]


class ColorOnlyRandAugment:
    def __init__(self, n=2, m=9):
        self.n = n  # 应用多少种操作
        self.m = m  # 操作的强度 (0-10)

    def __call__(self, img):
        # 随机选择n个不同的颜色操作
        ops = random.choices(COLOR_OPS, k=self.n)
        for op in ops:
            img = op(img, self.m)
        return img


# ===================================================================
# Mosaic 调度器类 (最终版)
# ===================================================================
class MosaicScheduler:
    """
    根据当前epoch动态计算mosaic_ratio。
    通过 step() 方法推进轮数，与PyTorch的LR调度器模式一致。
    """

    def __init__(self, logger, total_epochs, pretrain_epochs, finetune_epochs, high_ratio=0.75, low_ratio=0):
        if not all(isinstance(i, int) for i in [total_epochs, pretrain_epochs, finetune_epochs]):
            raise TypeError("Epoch counts must be integers.")
        if not total_epochs > pretrain_epochs + finetune_epochs:
            raise ValueError("Total epochs must be greater than the sum of pretrain and finetune epochs.")

        self.total_epochs = total_epochs
        self.pretrain_epochs = pretrain_epochs
        self.finetune_start_epoch = total_epochs - finetune_epochs + 1
        self.high_ratio = high_ratio
        self.low_ratio = low_ratio
        self.current_epoch = 0

        logger.info(
            f"MosaicScheduler initialized: Total Epochs={total_epochs}, "
            f"Pretrain (Ratio={low_ratio}) for {pretrain_epochs} epochs, "
            f"Finetune (Ratio={low_ratio}) starting at epoch {self.finetune_start_epoch}."
        )

    def step(self):
        self.current_epoch += 1

    def get_ratio(self):
        epoch_to_check = self.current_epoch + 1
        if epoch_to_check <= self.pretrain_epochs:
            return self.low_ratio
        elif epoch_to_check >= self.finetune_start_epoch:
            return self.low_ratio
        else:
            return self.high_ratio


# ===================================================================
# BaseDataset (包含所有增强构建块)
# ===================================================================
class BaseDataset(data.Dataset):
    def __init__(self, root):
        self.image_root_A = os.path.join(root, 'A')
        self.image_root_B = os.path.join(root, 'B')
        self.gt_root = os.path.join(root, 'label')

        self.images_A = sorted(
            [os.path.join(self.image_root_A, f) for f in os.listdir(self.image_root_A) if f.endswith(('.jpg', '.png'))])
        self.images_B = sorted(
            [os.path.join(self.image_root_B, f) for f in os.listdir(self.image_root_B) if f.endswith(('.jpg', '.png'))])
        self.gts = sorted(
            [os.path.join(self.gt_root, f) for f in os.listdir(self.gt_root) if f.endswith(('.jpg', '.png'))])
        self.filter_files()
        self.size = len(self.images_A)  # 定义数据集大小

    def filter_files(self):
        base_A = {os.path.splitext(os.path.basename(f))[0] for f in self.images_A}
        base_B = {os.path.splitext(os.path.basename(f))[0] for f in self.images_B}
        base_gt = {os.path.splitext(os.path.basename(f))[0] for f in self.gts}
        valid_basenames = base_A & base_B & base_gt
        assert len(valid_basenames) > 0, "没有找到三个目录共有的文件!"
        self.images_A = [f for f in self.images_A if os.path.splitext(os.path.basename(f))[0] in valid_basenames]
        self.images_B = [f for f in self.images_B if os.path.splitext(os.path.basename(f))[0] in valid_basenames]
        self.gts = [f for f in self.gts if os.path.splitext(os.path.basename(f))[0] in valid_basenames]
        self.images_A.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
        self.images_B.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
        self.gts.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
        assert len(self.images_A) == len(self.images_B) == len(self.gts), "筛选后文件数量仍不一致"

    def cv_random_flip(self, img_A, img_B, label):
        if random.random() > 0.5:
            return img_A.transpose(Image.FLIP_LEFT_RIGHT), img_B.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(
                Image.FLIP_LEFT_RIGHT)
        return img_A, img_B, label

    def random_crop_mosaic(self, image_A, image_B, label, crop_win_width, crop_win_height):
        image_width, image_height = image_A.size
        crop_win_width, crop_win_height = min(crop_win_width, image_width), min(crop_win_height, image_height)
        x_offset = (image_width - crop_win_width) // 2
        y_offset = (image_height - crop_win_height) // 2
        random_region = (x_offset, y_offset, x_offset + crop_win_width, y_offset + crop_win_height)
        return image_A.crop(random_region), image_B.crop(random_region), label.crop(random_region)

    def random_crop(self, image_A, image_B, label):
        border = 30
        image_width, image_height = image_A.size
        if image_width <= border or image_height <= border:
            return image_A, image_B, label
        crop_win_width = random.randint(image_width - border, image_width)
        crop_win_height = random.randint(image_height - border, image_height)
        x_offset = random.randint(0, image_width - crop_win_width)
        y_offset = random.randint(0, image_height - crop_win_height)
        random_region = (x_offset, y_offset, x_offset + crop_win_width, y_offset + crop_win_height)
        return image_A.crop(random_region), image_B.crop(random_region), label.crop(random_region)

    def random_rotation(self, image_A, image_B, label):
        if random.random() > 0.8:
            mode = Image.BICUBIC
            mask_mode = Image.NEAREST
            random_angle = random.randint(-15, 15)
            return image_A.rotate(random_angle, mode), image_B.rotate(random_angle, mode), label.rotate(random_angle,
                                                                                                        mask_mode)
        return image_A, image_B, label

    def color_enhance(self, image_A, image_B):
        enhanced_A = image_A
        enhanced_B = image_B
        ops = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color, ImageEnhance.Sharpness]
        for op in ops:
            factor_a = random.uniform(0.8, 1.2)
            factor_b = random.uniform(0.8, 1.2)
            enhanced_A = op(enhanced_A).enhance(factor_a)
            enhanced_B = op(enhanced_B).enhance(factor_b)
        return enhanced_A, enhanced_B

    def random_peper(self, img):
        img_arr = np.array(img)
        if img_arr.size == 0: return Image.fromarray(img_arr)
        noise_num = int(0.0015 * img_arr.size)
        rows, cols = img_arr.shape
        row_coords = np.random.randint(0, rows, size=noise_num)
        col_coords = np.random.randint(0, cols, size=noise_num)
        salt_num = noise_num // 2
        img_arr[row_coords[:salt_num], col_coords[:salt_num]] = 255
        img_arr[row_coords[salt_num:], col_coords[salt_num:]] = 0
        return Image.fromarray(img_arr)

    def load_img_and_mask(self, index):
        A = Image.open(self.images_A[index]).convert('RGB')
        B = Image.open(self.images_B[index]).convert('RGB')
        mask = Image.open(self.gts[index]).convert('L')
        return A, B, mask

    def load_mosaic_img_and_mask(self, index, trainsize):
        # other_indices = random.sample(range(self.size), 3)
        other_indices = random.sample([i for i in range(self.size) if i != index], 3)
        indexes = [index] + other_indices
        img_a_a, img_a_b, mask_a = self.load_img_and_mask(indexes[0])
        img_b_a, img_b_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c_a, img_c_b, mask_c = self.load_img_and_mask(indexes[2])
        img_d_a, img_d_b, mask_d = self.load_img_and_mask(indexes[3])
        w = trainsize;
        h = trainsize
        start_x = w // 4;
        strat_y = h // 4
        offset_x = random.randint(start_x, w - start_x)
        offset_y = random.randint(strat_y, h - strat_y)
        crop_size_a = (offset_x, offset_y);
        crop_size_b = (w - offset_x, offset_y);
        crop_size_c = (offset_x, h - offset_y);
        crop_size_d = (w - offset_x, h - offset_y)
        croped_a_a, croped_a_b, mask_crop_a = self.random_crop_mosaic(img_a_a, img_a_b, mask_a, crop_size_a[0],
                                                                      crop_size_a[1])
        croped_b_a, croped_b_b, mask_crop_b = self.random_crop_mosaic(img_b_a, img_b_b, mask_b, crop_size_b[0],
                                                                      crop_size_b[1])
        croped_c_a, croped_c_b, mask_crop_c = self.random_crop_mosaic(img_c_a, img_c_b, mask_c, crop_size_c[0],
                                                                      crop_size_c[1])
        croped_d_a, croped_d_b, mask_crop_d = self.random_crop_mosaic(img_d_a, img_d_b, mask_d, crop_size_d[0],
                                                                      crop_size_d[1])
        croped_a_a, croped_a_b, mask_crop_a = np.array(croped_a_a), np.array(croped_a_b), np.array(mask_crop_a)
        croped_b_a, croped_b_b, mask_crop_b = np.array(croped_b_a), np.array(croped_b_b), np.array(mask_crop_b)
        croped_c_a, croped_c_b, mask_crop_c = np.array(croped_c_a), np.array(croped_c_b), np.array(mask_crop_c)
        croped_d_a, croped_d_b, mask_crop_d = np.array(croped_d_a), np.array(croped_d_b), np.array(mask_crop_d)
        top_a = np.concatenate((croped_a_a, croped_b_a), axis=1);
        bottom_a = np.concatenate((croped_c_a, croped_d_a), axis=1);
        img_a = np.concatenate((top_a, bottom_a), axis=0)
        top_b = np.concatenate((croped_a_b, croped_b_b), axis=1);
        bottom_b = np.concatenate((croped_c_b, croped_d_b), axis=1);
        img_b = np.concatenate((top_b, bottom_b), axis=0)
        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1);
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1);
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        return Image.fromarray(np.ascontiguousarray(img_a)), Image.fromarray(
            np.ascontiguousarray(img_b)), Image.fromarray(np.ascontiguousarray(mask))


# ===================================================================
# Val_Dataset (无改动)
# ===================================================================
class Val_Dataset(BaseDataset):
    def __init__(self, root, trainsize, logger):
        super().__init__(root)
        self.trainsize = trainsize
        logger.info(f'验证集数量: {len(self.images_A)}')
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda mask: (mask > 0.5).float())])
        self.size = len(self.images_A)

    def __getitem__(self, index):
        image_A = self.load_img_and_mask(index)[0]
        image_B = self.load_img_and_mask(index)[1]
        gt = self.load_img_and_mask(index)[2]
        image_A = self.img_transform(image_A)
        image_B = self.img_transform(image_B)
        gt = self.gt_transform(gt)
        file_name = os.path.splitext(os.path.basename(self.images_A[index]))[0]
        return image_A, image_B, gt, file_name

    def __len__(self):
        return self.size


# ===================================================================
# FINAL REVISED IMPLEMENTATION
# ===================================================================

class SemiSupDataset(BaseDataset):
    """有标签数据集: 强增强 (含Mosaic) + TrivialAugment"""

    def __init__(self, root, trainsize, logger, mosaic_scheduler, train_ratio=0.2):
        super().__init__(root)
        self.trainsize = trainsize
        self.mosaic_scheduler = mosaic_scheduler
        num_total = len(self.images_A)
        self.labeled_indices = random.sample(range(num_total), int(num_total * train_ratio))
        self.size = len(self.labeled_indices)
        logger.info(f'有标签数据集: {self.size} / {num_total}')

        # 将 TrivialAugmentWide 整合到最终的 transform 中
        self.final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda mask: (mask > 0.5).float())])

        # self.auto_augment = TrivialAugmentWide()
        # self.auto_augment = ColorOnlyRandAugment(n = 2, m = 7)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        original_index = self.labeled_indices[index]

        # --- 新增: 获取文件名 ---
        file_name = os.path.splitext(os.path.basename(self.images_A[original_index]))[0]

        # 1. 结构性变换 (Mosaic, Crop, Flip)
        current_mosaic_ratio = self.mosaic_scheduler.get_ratio()
        # 2. 根据课程阶段，选择增强策略
        if current_mosaic_ratio == 0:
            # --- 阶段A: 预热/微调期 ---
            image_A, image_B, gt = self.load_img_and_mask(original_index)
            image_A, image_B, gt = self.cv_random_flip(image_A, image_B, gt)
            image_A, image_B, gt = self.random_crop(image_A, image_B, gt)
            image_A, image_B, gt = self.random_rotation(image_A, image_B, gt)
            image_A, image_B = self.color_enhance(image_A, image_B)
        else:
            # --- 阶段B: 核心训练期 ---
            if random.random() < current_mosaic_ratio:
                # 分支B1: 结构复杂
                image_A, image_B, gt = self.load_mosaic_img_and_mask(original_index, self.trainsize)
                image_A, image_B, gt = self.cv_random_flip(image_A, image_B, gt)
                image_A, image_B, gt = self.random_rotation(image_A, image_B, gt)
                image_A, image_B = self.color_enhance(image_A, image_B)
            else:
                # 分支B2: 结构简单
                image_A, image_B, gt = self.load_img_and_mask(original_index)
                image_A, image_B, gt = self.cv_random_flip(image_A, image_B, gt)
                image_A, image_B, gt = self.random_crop(image_A, image_B, gt)
                image_A, image_B, gt = self.random_rotation(image_A, image_B, gt)
                image_A, image_B = self.color_enhance(image_A, image_B)

        gt = self.random_peper(gt)

        # 3. 尺寸调整和最终转换
        image_A = image_A.resize((self.trainsize, self.trainsize), Image.BILINEAR)
        image_B = image_B.resize((self.trainsize, self.trainsize), Image.BILINEAR)

        image_A_s = self.final_transform(image_A)
        image_B_s = self.final_transform(image_B)
        gt_s = self.gt_transform(gt)

        # 返回统一格式的元组
        return image_A_s, image_B_s, image_A_s, image_B_s, gt_s, True, file_name


class SemiUnsupeDataset(BaseDataset):
    """无标签数据集: 单图弱-强增强对"""

    def __init__(self, root, trainsize, logger, mosaic_scheduler, sup_indices, is_vis=False):
        super().__init__(root)
        self.trainsize = trainsize
        # 共享同一个调度器实例
        self.mosaic_scheduler = mosaic_scheduler

        num_total = len(self.images_A)
        self.unlabeled_indices = sorted(list(set(range(num_total)) - set(sup_indices)))
        self.size = len(self.unlabeled_indices)
        logger.info(f'无标签数据集 (终极完美版): {self.size} / {num_total}')

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda mask: (mask > 0.5).float())])

        # self.auto_augment = TrivialAugmentWide()
        # self.auto_augment = ColorOnlyRandAugment(n = 2, m = 7)
        self.color_enhance = self.color_enhance
        self.is_vis = is_vis

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        original_index = self.unlabeled_indices[index]

        # --- 新增: 获取文件名 ---
        file_name = os.path.splitext(os.path.basename(self.images_A[original_index]))[0]

        # =======================================================
        # 修改开始：可视化模式专用通道 (纯净读取)
        # =======================================================
        if self.is_vis:
            # 1. 基础读取
            image_A_base, image_B_base, gt_base = self.load_img_and_mask(original_index)
            
            # 2. 仅做 Resize (不做 Flip, Crop, Rotation)
            # 必须保证 resize 到 trainsize，否则 Tensor 维度对不上
            image_A_base = image_A_base.resize((self.trainsize, self.trainsize), Image.BILINEAR)
            image_B_base = image_B_base.resize((self.trainsize, self.trainsize), Image.BILINEAR)
            
            # 3. 准备输出 (后续代码会处理 ToTensor)
            # 为了兼容后面的代码逻辑，直接赋值给 base 变量
            
            # 注意：下面的 color_enhance 也要跳过或者保持弱增强
            # 这里我们为了可视化原始效果，建议也不做 color_enhance
            image_A_w_pil = image_A_base
            image_B_w_pil = image_B_base
            image_A_s_pil = image_A_base # 强增强视图也保持原样
            image_B_s_pil = image_B_base 

            # 4. 转 Tensor
            image_A_w = self.base_transform(image_A_w_pil)
            image_B_w = self.base_transform(image_B_w_pil)
            image_A_s = self.base_transform(image_A_s_pil)
            image_B_s = self.base_transform(image_B_s_pil)
            gt_s = self.gt_transform(gt_base)

            return image_A_w, image_B_w, image_A_s, image_B_s, gt_s, False, file_name
        # =======================================================
        # 修改结束
        # =======================================================

        # --- 1. 从共享的调度器获取当前训练节奏 ---
        current_mosaic_ratio = self.mosaic_scheduler.get_ratio()

        # --- 2. 生成基础几何变换视图 (节奏与有标签数据完全同步) ---
        if random.random() < current_mosaic_ratio:
            # 分支A: 核心训练期 (高概率)
            # 基础视图是Mosaic图, 强视图用温和外观变换
            image_A_base, image_B_base, gt_base = self.load_mosaic_img_and_mask(original_index, self.trainsize)
            image_A_base, image_B_base, gt_base = self.cv_random_flip(image_A_base, image_B_base, gt_base)
            image_A_base, image_B_base, gt_base = self.random_rotation(image_A_base, image_B_base, gt_base)
        else:
            # 分支B: 预热/微调期, 或核心训练期的低概率情况
            # 基础视图是单图
            image_A_base, image_B_base, gt_base = self.load_img_and_mask(original_index)
            image_A_base, image_B_base, gt_base = self.cv_random_flip(image_A_base, image_B_base, gt_base)
            image_A_base, image_B_base, gt_base = self.random_crop(image_A_base, image_B_base, gt_base)
            image_A_base, image_B_base, gt_base = self.random_rotation(image_A_base, image_B_base, gt_base)

        # --- 3. 生成最终的弱/强视图 ---
        image_A_w_pil = image_A_base
        image_B_w_pil = image_B_base
        # image_A_w_pil, image_B_w_pil = self.color_enhance(image_A_base.copy(), image_B_base.copy())
        image_A_s_pil, image_B_s_pil = self.color_enhance(image_A_base.copy(), image_B_base.copy())

        # --- 4. 尺寸调整和最终转换 ---
        image_A_w = image_A_w_pil.resize((self.trainsize, self.trainsize), Image.BILINEAR)
        image_B_w = image_B_w_pil.resize((self.trainsize, self.trainsize), Image.BILINEAR)
        image_A_w = self.base_transform(image_A_w)
        image_B_w = self.base_transform(image_B_w)

        image_A_s = image_A_s_pil.resize((self.trainsize, self.trainsize), Image.BILINEAR)
        image_B_s = image_B_s_pil.resize((self.trainsize, self.trainsize), Image.BILINEAR)
        image_A_s = self.base_transform(image_A_s)
        image_B_s = self.base_transform(image_B_s)

        gt_s = self.gt_transform(gt_base)

        return image_A_w, image_B_w, image_A_s, image_B_s, gt_s, False, file_name


# ===================================================================
# 辅助函数 (最终版)
# ===================================================================
def get_semi_loader(root, batchsize, trainsize, train_ratio, logger, mosaic_scheduler, worker_init_fn=None,
                    num_workers=4, shuffle=True, pin_memory=True):
    sup_dataset = SemiSupDataset(root, trainsize, logger, mosaic_scheduler, train_ratio)
    unsup_dataset = SemiUnsupeDataset(root, trainsize, logger, mosaic_scheduler, sup_dataset.labeled_indices)

    combined_dataset = torch.utils.data.ConcatDataset([sup_dataset, unsup_dataset])
    data_loader = data.DataLoader(dataset=combined_dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  worker_init_fn=worker_init_fn,
                                  pin_memory=pin_memory)
    # 返回数据集实例，以便在需要时访问（虽然在新设计中不一定需要）
    # return data_loader, sup_dataset, unsup_dataset
    return data_loader


def get_val_loader(root, batchsize, trainsize, logger, num_workers=1, shuffle=False, pin_memory=True):
    dataset = Val_Dataset(root, trainsize, logger)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def gen_label(root, trainsize, train_ratio, logger):
    # 此函数仅用于生成文件列表，不需要动态mosaic，可使用固定ratio或dummy scheduler
    dummy_scheduler = MosaicScheduler(100, 1, 1, high_ratio=0.0)
    sup_dataset = SemiSupDataset(root, trainsize, logger, dummy_scheduler, train_ratio)

    file_list = sup_dataset.images_A
    labeled_indices = sup_dataset.labeled_indices
    labeled_files = [os.path.basename(file_list[i]) for i in labeled_indices]
    unlabeled_files = [os.path.basename(f) for i, f in enumerate(file_list) if i not in labeled_indices]

    list_dir = '../data/LEVIR-CD-256/list'
    os.makedirs(list_dir, exist_ok=True)

    ratio_str = str(int(train_ratio * 100))
    supervised_path = os.path.join(list_dir, f"{ratio_str}_train_supervised.txt")
    unsupervised_path = os.path.join(list_dir, f"{ratio_str}_train_unsupervised.txt")

    with open(supervised_path, "w") as f:
        f.write("\n".join(labeled_files))
    with open(unsupervised_path, "w") as f:
        f.write("\n".join(unlabeled_files))

    logger.info(f"生成监督文件列表：{supervised_path} ({len(labeled_files)} 样本)")
    logger.info(f"生成无监督文件列表：{unsupervised_path} ({len(unlabeled_files)} 样本)")
