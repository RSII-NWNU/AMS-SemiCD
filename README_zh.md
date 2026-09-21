# AMS-SemiCD

**English：[README.md](README.md)**

论文 **《AMS-SemiCD: A Synergistic Framework for Semi-Supervised Change Detection via an Adaptive Pseudo-Labeling Strategy and a Multi-Branch Network》** 的代码说明与复现实验指南。

## 项目简介

半监督变化检测通过少量有标签样本和大量无标签样本，降低像素级标注成本。AMS-SemiCD 针对两个互补问题进行设计：类别不平衡导致的伪标签置信度分布差异，以及双时相遥感图像中局部细节与长距离上下文难以同时建模的问题。

框架包括：

- **ASPR（Adaptive Statistical Pseudo-Label Refinement）**：跟踪类别级置信度统计量，计算类别自适应动态阈值，并为教师模型预测分配贡献自适应的软权重。
- **MSE-Net（Multi-branch Synergistic Enhancement Network）**：融合 CNN 局部细节分支和 VMamba 全局上下文分支。
- **MSSF**：多分支时空协同融合模块，融合局部和全局特征。
- **MSDB**：多分支时空差异增强模块，突出双时相差异信息。
- **AGAR**：自适应门控注意力细化模块，用于多层特征重建与融合。

训练采用 Mean-Teacher 学生-教师框架。初始监督阶段使用有标签样本训练 Student，并初始化 ASPR 统计量；半监督阶段由 Teacher 处理弱增强无标签样本，ASPR 对伪标签置信度进行细化，再使用强增强样本训练 Student。

论文在 **GoogleGZ-CD、WHU-CD 和 LEVIR-CD** 三个数据集上进行实验。论文报告在 LEVIR-CD 仅使用 20% 标签时，F1 为 91.17%，IoU 为 83.78%。

## 方法结构图

### 训练框架

![AMS-SemiCD 训练框架](images/ams.png)

### MSE-Net 网络结构

![MSE-Net 网络结构](images/mse.png)

## 仓库结构

```text
.
├── images/
│   ├── ams.png                         # 训练框架图
│   └── mse.png                         # MSE-Net 结构图
├── model/
│   ├── model.py                        # 正式 SemiModel / SemiModel_ALL
│   ├── mssf.py                         # MSSF
│   ├── msdb.py                         # MSDB
│   ├── agar.py                         # AGAR 与注意力模块
│   └── vmamba/                         # VMamba 主干和 selective scan
├── utils/
│   ├── data_loader_new.py              # 数据加载与增强
│   └── adaptive_statistica_pseudolabel_refinement.py
├── msce_train.py                       # 训练入口
├── msce_test.py                        # 检查点测试入口
├── msce_visual.py                      # 预测与特征可视化
├── requirements.txt
└── model/vmamba/vmambaweight/
    └── vssm_small_0229_ckpt_epoch_222.pth
```

当前训练、测试和可视化入口只使用正式的 `SemiModel_ALL` 模型，不支持已删除的消融模型入口。

## 环境安装

已验证环境为 Linux、Python 3.10.19、PyTorch 2.6.0（CUDA 12.4）和单张 NVIDIA GPU。

```bash
conda create -n ams-semicd python=3.10.19 -y
conda activate ams-semicd

# 先安装 CUDA 12.4 对应的 PyTorch。
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 安装其余 Python 依赖。
python -m pip install -r requirements.txt
```

检查安装结果：

```bash
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### selective-scan CUDA 扩展

VMamba 使用的 `selective-scan` CUDA 扩展源码位于项目内。确保 PyTorch、CUDA Toolkit 和 C++/CUDA 编译器可用后执行：

```bash
cd model/vmamba/kernels/selective_scan
python setup.py install
cd ../../../../
```

该扩展与当前 Python、PyTorch、CUDA Toolkit 和 GPU 架构相关，其他机器编译的 `.so` 文件不一定可以直接复用。缺少扩展时部分路径可以回退到较慢的 PyTorch 实现，但复现实验建议完成 CUDA 编译。

## 数据集下载与准备

### 下载地址

项目使用的数据集可通过提供的百度网盘分享获取：

- **项目数据包：**[百度网盘：开源数据集](https://pan.baidu.com/s/1wBIEjbyeRCQuueCT5C_DZg?pwd=65x3)
- **提取码：**`65x3`

官方/原始数据集来源：

- [LEVIR-CD 官方项目页面](https://justchenhao.github.io/LEVIR/)
- [WHU Building / WHU-CD 官方页面](https://gpcv.whu.edu.cn/data/building_dataset.html)
- [GZ-CD 原始项目仓库](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)

请遵循原始数据集提供方的学术使用、署名和图像再分发要求。百度网盘仅作为便利下载入口，数据集说明和使用条款以原始提供方为准。

### 数据目录结构

数据加载器要求每个数据集的每个划分都有 `A`、`B`、`label` 三个目录，且对应文件的文件名主体一致：

```text
data/
├── LEVIR-CD-256/
│   ├── train/{A,B,label}/
│   ├── val/{A,B,label}/
│   └── test/{A,B,label}/
├── WHU-CD-256/
│   ├── train/{A,B,label}/
│   ├── val/{A,B,label}/
│   └── test/{A,B,label}/
└── GZ-CD-256/
    ├── train/{A,B,label}/
    ├── val/{A,B,label}/
    └── test/{A,B,label}/
```

例如：

```text
train/A/000001.png
train/B/000001.png
train/label/000001.png
```

默认输入尺寸为 256 × 256。标签缩放和旋转使用最近邻插值，转换为张量后再次二值化，因此 mask 保持二值。

### 数据根目录配置

当前 `msce_train.py` 将三个数据集映射到以下服务器路径：

```text
/data/proj/ypc/ams-new/data/LEVIR-CD-256/
/data/proj/ypc/ams-new/data/WHU-CD-256/
/data/proj/ypc/ams-new/data/GZ-CD-256/
```

在其他机器上训练时，请将数据放到这些路径，或修改 `msce_train.py` 中三个 `train_root` / `val_root` 赋值为本机路径。测试和可视化入口可以直接使用 `--test_root` 指定测试集路径。

## 预训练权重

默认 VMamba 权重路径为：

```text
model/vmamba/vmambaweight/vssm_small_0229_ckpt_epoch_222.pth
```

VMamba 预训练权重可从官方 release 下载：

- [vssm_small_0229_ckpt_epoch_222.pth](https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth)

如果文件不存在，请下载对应权重并放到该路径，或通过参数指定其他路径：

```bash
--vmamba_weight_path /path/to/vssm_small_0229_ckpt_epoch_222.pth
```

ResNeXt-50 分支默认使用 torchvision 预训练权重（`res_pretrained=True`）。首次构建模型时，torchvision 可能会自动下载并缓存权重。若不使用该权重，添加：

```bash
--no_res_pretrained
```

## 训练

请在仓库根目录运行。下面示例使用 WHU-CD、5% 有标签数据训练：

```bash
python msce_train.py \
  --data_name WHU \
  --train_ratio 0.05 \
  --gpu_id 0 \
  --epoch 105 \
  --batchsize 8 \
  --trainsize 256 \
  --vmamba_weight_path ./model/vmamba/vmambaweight/vssm_small_0229_ckpt_epoch_222.pth
```

其他支持的数据集：

```bash
python msce_train.py --data_name LEVIR --train_ratio 0.20 --gpu_id 0
python msce_train.py --data_name GZ --train_ratio 0.20 --gpu_id 0
```

当前入口只支持 `SemiModel_ALL`。ASPR 默认启用；如果只想关闭 ASPR 监控输出而不关闭 ASPR 本身，可执行：

```bash
python msce_train.py --data_name WHU --disable_aspr_monitor
```

训练结果、检查点、日志和 TensorBoard 文件会写入 `--save_path` 下自动生成的实验目录，例如：

```text
output/C2F-SemiCD/WHU/SemiModel_ALL_WHU_0.05_<timestamp>/
```

断点续训时，添加 `--model_load`，并将 `--load_path` 设置为已有实验目录。

## 测试

`msce_test.py` 会扫描 `--save_path` 下的检查点，并使用正式 `SemiModel_ALL` 模型：

```bash
python msce_test.py \
  --test_root /path/to/WHU-CD-256/test \
  --save_path ./output/C2F-SemiCD/WHU/SemiModel_ALL_WHU_0.05_<timestamp> \
  --gpu_id 0
```

测试集同样需要 `A`、`B`、`label` 目录结构。CUDA 不可用时，`msce_test.py` 会自动回退到 CPU。

## 可视化

使用 `msce_visual.py` 生成预测图、混淆图、置信度热图和可选的特征可视化：

```bash
python msce_visual.py \
  --dataset_name WHU \
  --test_root /path/to/WHU-CD-256/test \
  --save_path ./output/C2F-SemiCD/WHU/SemiModel_ALL_WHU_0.05_<timestamp> \
  --save_dir ./test_results/WHU-SemiModel_ALL \
  --checkpoint_type auto \
  --gpu_id 0
```

`--checkpoint_type` 支持 `student`、`teacher` 和 `auto`；`auto` 会根据检查点信息选择对应权重。

## 引用

如果使用本代码或 AMS-SemiCD 方法，请引用论文：

```bibtex
@article{zhang2026amssemicd,
  title   = {AMS-SemiCD: A Synergistic Framework for Semi-Supervised Change Detection via an Adaptive Pseudo-Labeling Strategy and a Multi-Branch Network},
  author  = {Di Zhang and Peicheng Yue and Huifang Ma and Zhanjun Hao and Xin He and Yun Liu and Jiaqi Zhao},
  journal = {IEEE Transactions on Geoscience & Remote Sensing},
  year    = {2026}
}
```

论文题目为 *AMS-SemiCD: A Synergistic Framework for Semi-Supervised Change Detection via an Adaptive Pseudo-Labeling Strategy and a Multi-Branch Network*；代码仓库为 [RSII-NWNU/AMS-SemiCD](https://github.com/RSII-NWNU/AMS-SemiCD)。最终接收版本论文随项目材料提供。

## 注意事项

- 本仓库不重新分发原始遥感数据，请从上面的数据源获取并遵守数据集条款。
- 当前代码只维护正式 `SemiModel_ALL` 路径。
- 默认命令面向 Linux + CUDA 环境；CPU 适合冒烟测试，不适合完整训练。
- 编译 `selective-scan` 前请确认 NVIDIA 驱动和 CUDA Toolkit 版本匹配。
