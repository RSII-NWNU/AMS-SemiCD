import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d, DeformConv2d
from .Auxiliarymodule import CoordAtt, FlashCrossAttention, MultiScaleSpatialAtt


# 统一对齐层（采用可变形卷积实现）
class UnifiedAlignmentV1(nn.Module):
    def __init__(self, c):
        super().__init__()
        # 偏移量生成网络
        self.offset_net = nn.Sequential(
            nn.Conv2d(2 * c, c, 3, padding=1),
            nn.InstanceNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, 2 * 3 * 3, 1)  # 输出18通道偏移量（3x3卷积核）
        )
        # 可变形卷积
        self.deform_conv = DeformConv2d(
            c, c, kernel_size=3, padding=1
        )

    def forward(self, x_ref, x_tar):
        # 拼接参考帧和目标帧作为输入
        offset = self.offset_net(torch.cat([x_ref, x_tar], dim=1))
        return self.deform_conv(x_tar, offset)


class UnifiedAlignmentV2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.offset_net = nn.Sequential(
            # Stage 1: 基础卷积层（保持与原始版本一致）
            nn.Conv2d(2 * c, c, 3, padding=1),
            # 新增归一化层（关键位置1：缓解协变量偏移）
            nn.InstanceNorm2d(c),  # 计算量可忽略，无参数
            nn.GELU(),

            # Stage 2: 轻量残差块（深度可分离卷积）
            self._make_dw_block(c),

            # Stage 3: 输出层（保持原始结构）
            nn.Conv2d(c, 2 * 3 * 3, 1)
        )
        self.deform_conv = DeformConv2d(c, c, kernel_size=3, padding=1)
        self._init_weights()

    def _make_dw_block(self, channels):
        """带归一化的深度可分离卷积块"""
        return nn.Sequential(
            # 深度卷积
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            # 关键位置2：卷积后归一化
            nn.InstanceNorm2d(channels),  # 无参数
            nn.GELU(),

            # 逐点卷积
            nn.Conv2d(channels, channels, 1),
            # 关键位置3：残差连接前归一化
            nn.InstanceNorm2d(channels),
            nn.GELU()

        )

    def _init_weights(self):
        """保持零初始化约束"""
        nn.init.constant_(self.offset_net[-1].weight, 0)
        nn.init.constant_(self.offset_net[-1].bias, 0)

    def forward(self, x_ref, x_tar):
        offset = torch.tanh(
            self.offset_net(torch.cat([x_ref, x_tar], dim=1))
        )
        return self.deform_conv(x_tar, offset)


# 增强型像素差异模块
class EnhancedDifference(nn.Module):
    def __init__(self, c):
        super().__init__()

        # 协调注意力机制（CoordAttention改进版）
        self.coord_attn = CoordAtt(inp=c, oup=c, reduction=16)

        # 残差连接适配器
        self.res_adapt = nn.Conv2d(c, c, 1) if c != c else nn.Identity()

    def forward(self, x_ref, x_aligned):
        # 差异特征生成（含绝对值增强）
        diff = torch.abs(x_ref - x_aligned)  # 绝对值处理

        # 协调注意力调制
        aw, ah = self.coord_attn(diff)

        # 残差增强输出
        return diff * aw * ah


class EnhancedCrossAttnV1(nn.Module):
    def __init__(self, c: int, num_heads: int = 4):
        super().__init__()
        # 参数校验
        assert c % (4 * num_heads) == 0, f"通道数c必须能被4*num_heads整除，当前c={c}, num_heads={num_heads}"

        self.num_heads = num_heads
        self.head_dim = c // 4 // num_heads  # 计算每个头的维度

        # 多头投影层
        # self.query = nn.Conv2d(c, c // 4, 3, padding = 1)
        # self.key = nn.Conv2d(c, c // 4, 3, padding = 1)
        # self.value = nn.Conv2d(c, c, 3, padding = 1)

        # 上下文增强网络
        # self.ctx_net = nn.Sequential(
        #     nn.Conv2d(c, c, 3, padding=1, groups=4),
        #     nn.GELU(),
        #     nn.Conv2d(c, c, 1),
        # )
        self.ctx_net = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, padding=1, groups=2),  # 通道压缩
            nn.GELU(),  # 保留部分负值信息
            nn.Conv2d(c // 2, c, 1),  # 通道恢复
            nn.Sigmoid(),  # 门控机制
            nn.Dropout2d(0.1)  # 防止过拟合
        )
        self.common_attn = FlashCrossAttention(c, num_heads=8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_ref: torch.Tensor, x_aligned: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x_ref.shape

        # 生成Q/K/V（保持Q来自x_ref，K/V来自x_aligned）
        # Q = self.query(x_ref)
        # K = self.key(x_aligned)
        # V = self.value(x_aligned)

        # 重塑形状适配Flash Attention
        B, C, H, W = x_ref.shape
        f_common = torch.cat([x_ref, x_aligned], dim=0)  # [2B, C, H, W]
        f_common = f_common.permute(0, 2, 3, 1).view(2 * B, H * W, C)  # [2B, L, C]

        # 交叉注意力计算（q=f_res, k/v=f_vm_aligned）
        attn_out = self.common_attn(
            query=f_common[:B],  # 使用f_res作为query
            key=f_common[B:],  # 使用f_vm_aligned作为key
            value=f_common[B:]  # 使用f_vm_aligned作为value
        )  # [B, L, C]

        # 恢复原始形状
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, H, W)

        # 关键修改：用Xt2减去注意力结果
        return self.ctx_net(x_aligned - attn_out)

    def _fallback_forward(self, x_ref: torch.Tensor, x_aligned: torch.Tensor) -> torch.Tensor:
        """备用实现（同步修改残差方向）"""
        B, _, H, W = x_ref.shape

        # 原始注意力计算
        Q = self.query(x_ref).view(B, -1, H * W).permute(0, 2, 1)
        K = self.key(x_aligned).view(B, -1, H * W)
        V = self.value(x_aligned).view(B, -1, H * W)

        # 注意力计算
        attn = torch.softmax(torch.bmm(Q, K), dim=-1)
        attn_out = torch.bmm(V, attn.permute(0, 2, 1))

        # 关键修改：恢复形状后执行减法
        attn_out = attn_out.view(B, -1, H, W)
        return self.ctx_net(x_aligned - attn_out)


class EnhancedCrossAttnV2(nn.Module):
    def __init__(self, c: int, num_heads: int = 4):
        super().__init__()
        # 参数校验
        assert c % (4 * num_heads) == 0, f"通道数c必须能被4*num_heads整除，当前c={c}, num_heads={num_heads}"

        self.num_heads = num_heads
        self.head_dim = c // 4 // num_heads  # 计算每个头的维度

        self.ctx_net = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, padding=1, groups=2),  # 通道压缩
            nn.GELU(),  # 保留部分负值信息
            nn.Conv2d(c // 2, c, 1),  # 通道恢复
            nn.Sigmoid(),  # 门控机制
            nn.Dropout2d(0.1)  # 防止过拟合
        )
        self.common_attn = FlashCrossAttention(c, num_heads=8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_ref: torch.Tensor, x_aligned: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x_ref.shape

        # 重塑形状适配Flash Attention
        B, C, H, W = x_ref.shape
        f_common = torch.cat([x_ref, x_aligned], dim=0)  # [2B, C, H, W]
        f_common = f_common.permute(0, 2, 3, 1).view(2 * B, H * W, C)  # [2B, L, C]

        # 交叉注意力计算（q=f_res, k/v=f_vm_aligned）
        attn_out = self.common_attn(
            query=f_common[:B],  # 使用f_res作为query
            key=f_common[B:],  # 使用f_vm_aligned作为key
            value=f_common[B:]  # 使用f_vm_aligned作为value
        )  # [B, L, C]

        # 恢复原始形状
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, H, W)
        attn_out = 1 - self.sigmoid(attn_out)
        return x_aligned * attn_out


class FeatureEnhancementModuleV1(nn.Module):
    def __init__(self, c):
        super().__init__()
        # 对齐层
        # self.aligner = UnifiedAlignmentV2(c)

        # 双分支结构
        self.diff_branch = EnhancedDifference(c)
        self.attn_branch = EnhancedCrossAttnV2(c)

        # 动态融合
        self.fusion_net = nn.Sequential(
            nn.Conv2d(2 * c, c // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c // 2, 2, 3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        # 特征对齐
        # x2_aligned = self.aligner(x1, x2)

        # 双分支处理
        diff_feat = self.diff_branch(x1, x2)
        attn_feat = self.attn_branch(x1, x2)

        # 自适应融合
        weights = self.fusion_net(torch.cat([diff_feat, attn_feat], dim=1))
        return weights[:, 0:1] * diff_feat + weights[:, 1:2] * attn_feat


class MSDB(nn.Module):
    def __init__(self, c):
        super().__init__()
        # 恢复对齐层
        self.aligner = UnifiedAlignmentV2(c)

        # 双分支结构
        self.diff_branch = EnhancedDifference(c)
        self.mssa_branch = MultiScaleSpatialAtt(c)

        # 简化融合层
        self.fusion_atn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, 1, 1),
            nn.Sigmoid()  # 独立权重学习
        )

    def forward(self, x1, x2):
        # 特征对齐
        x2_aligned = self.aligner(x1, x2)

        # 双分支处理
        diff_feat = self.diff_branch(x1, x2_aligned)

        attn_feat = self.mssa_branch(x1, x2_aligned)  # 输入对齐特征

        # 动态融合（权重基于特征重要性）
        w_diff = self.fusion_atn(diff_feat)
        w_attn = 1 - w_diff  # 互补权重
        return w_diff * diff_feat + w_attn * attn_feat


class FeatureEnhancementModuleV3(nn.Module):
    def __init__(self, c):
        super().__init__()
        # 恢复对齐层
        self.aligner = UnifiedAlignmentV2(c)

        # 双分支结构
        self.diff_branch = EnhancedDifference(c)
        self.mssa_branch = MultiScaleSpatialAtt(c)

        # 动态融合
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))


    def forward(self, x1, x2):
        # 特征对齐
        x2_aligned = self.aligner(x1, x2)

        # 双分支处理
        diff_feat = self.diff_branch(x1, x2_aligned)

        attn_feat = self.mssa_branch(x1, x2_aligned)  # 输入对齐特征
        total = torch.sigmoid(self.alpha) + torch.sigmoid(self.beta) + 1e-6
        w1 = torch.sigmoid(self.alpha) / total
        w2 = torch.sigmoid(self.beta) / total
        # 自适应融合
        return w1 * diff_feat + w2 * attn_feat
