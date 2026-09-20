import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from .Auxiliarymodule import FlashCrossAttention, CoordAtt


# 特征对齐与公共特征提取
# 对齐两分支的通道和空间维度，并提取共享语义信息。
class FeatureAligner(nn.Module):
    def __init__(self, c_res, c_vm, out_c):
        super().__init__()
        # 可变形对齐参数
        # self.offset_conv = nn.Conv2d(2 * out_c, 2 * 3 * 3, 3, padding = 1)
        # 通道对齐
        self.conv_res = nn.Conv2d(c_res, out_c, 1) if c_res != out_c else nn.Identity()
        self.conv_vm = nn.Conv2d(c_vm, out_c, 1) if c_vm != out_c else nn.Identity()
        # 使用Flash Attention v2的交叉注意力
        self.common_attn = FlashCrossAttention(out_c, num_heads = 8)

    def forward(self, f_res, f_vm):
        # Step 1: 通道对齐
        f_res = self.conv_res(f_res)  # [B, C, H, W]
        f_vm = self.conv_vm(f_vm)

        # Step 2: 可变形空间对齐
        # offset = self.offset_conv(torch.cat([f_res, f_vm], dim = 1))
        # f_vm_aligned = deform_conv2d(
        #     input = f_vm,
        #     offset = offset,
        #     weight = nn.Parameter(torch.ones(1, f_vm.size(1), 3, 3)),
        #     padding = 1
        # )

        # Step 3: 交叉注意力融合
        B, C, H, W = f_res.shape
        f_common = torch.cat([f_res, f_vm], dim = 0)  # [2B, C, H, W]
        f_common = f_common.permute(0, 2, 3, 1).view(2 * B, H * W, C)  # [2B, L, C]

        # 交叉注意力计算（q=f_res, k/v=f_vm_aligned）
        attn_out = self.common_attn(
            query = f_common[:B],  # 使用f_res作为query
            key = f_common[B:],  # 使用f_vm_aligned作为key
            value = f_common[B:]  # 使用f_vm_aligned作为value
        )  # [B, L, C]

        # 恢复原始形状
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, H, W)

        return f_res, f_vm, attn_out


# ----------------- 优化后的子模块 -----------------
class ChannelInteractionV1(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        # 参数共享的投影层
        # self.proj = nn.Conv2d(out_c, out_c, 1)
        # ResNeXt→VMamba 和 VMamba→ResNeXt 使用独立投影层
        self.proj_res = nn.Conv2d(out_c, out_c, 1)  # ResNeXt专用
        self.proj_vm = nn.Conv2d(out_c, out_c, 1)  # VMamba专用
        self.v_proj = nn.Conv2d(2 * out_c, out_c, 1)

    def forward(self, f_a, f_b, direction):
        B, C, H, W = f_a.shape
        L = H * W

        # 动态生成Q/K
        # direction参数控制投影方向
        Q = self.proj_res(f_a).view(B, C, L) if direction == 'res2vm' else self.proj_vm(f_a).view(B, C, L)  # [B, C, L]
        K = self.proj_vm(f_b).view(B, C, L) if direction == 'res2vm' else self.proj_res(f_b).view(B, C, L)
        V = self.v_proj(torch.cat([f_a, f_b], dim = 1)).view(B, C, L)

        # 高效通道注意力
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)), dim = -1)  # [B, C, C]
        weighted = torch.bmm(attn, V).view(B, C, H, W)
        return f_b + weighted  # 残差连接


class ChannelInteractionV2(nn.Module):
    def __init__(self, out_c, groups = 8):
        super().__init__()
        # 使用分组卷积减少参数
        self.proj_res = nn.Conv2d(out_c, out_c, 1, groups = groups)
        self.proj_vm = nn.Conv2d(out_c, out_c, 1, groups = groups)
        # v_proj同样使用分组卷积
        self.v_proj = nn.Conv2d(2 * out_c, out_c, 1, groups = groups)

    def forward(self, f_a, f_b, direction):
        B, C, H, W = f_a.shape
        L = H * W

        # 动态生成Q/K
        # direction参数控制投影方向
        Q = self.proj_res(f_a).view(B, C, L) if direction == 'res2vm' else self.proj_vm(f_a).view(B, C, L)  # [B, C, L]
        K = self.proj_vm(f_b).view(B, C, L) if direction == 'res2vm' else self.proj_res(f_b).view(B, C, L)
        V = self.v_proj(torch.cat([f_a, f_b], dim = 1)).view(B, C, L)

        # 高效通道注意力
        # attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)), dim = -1)  # [B, C, C]
        # weighted = torch.bmm(attn, V).view(B, C, H, W)
        # 修正1：增加缩放因子（关键修改）
        scale_factor = 1.0 / (L ** 0.5)  # 空间维度归一化
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * scale_factor

        # 修正2：稳定数值计算（可选）
        attn = F.softmax(attn_logits, dim = -1)  # [B, C, C]

        weighted = torch.bmm(attn, V).view(B, C, H, W)
        return f_b + weighted  # 残差连接


class ChannelInteractionV3(nn.Module):
    def __init__(self, out_c, groups = 8):
        super().__init__()

        # 动态投影层（增加层归一化）
        self.proj_res = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, groups = groups),
            nn.GroupNorm(groups, out_c)
        )
        self.proj_vm = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, groups = groups),
            nn.GroupNorm(groups, out_c)
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(2 * out_c, out_c, 1, groups = groups),
            nn.GELU()
        )

    def forward(self, f_a, f_b, direction):
        B, C, H, W = f_a.shape
        L = H * W

        # 动态生成Q/K
        # direction参数控制投影方向
        Q = self.proj_res(f_a).view(B, C, L) if direction == 'res2vm' else self.proj_vm(f_a).view(B, C, L)  # [B, C, L]
        K = self.proj_vm(f_b).view(B, C, L) if direction == 'res2vm' else self.proj_res(f_b).view(B, C, L)
        V = self.v_proj(torch.cat([f_a, f_b], dim = 1)).view(B, C, L)

        # 高效通道注意力
        # attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)), dim = -1)  # [B, C, C]
        # weighted = torch.bmm(attn, V).view(B, C, H, W)
        # 修正1：增加缩放因子（关键修改）
        scale_factor = 1.0 / (L ** 0.5)  # 空间维度归一化
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * scale_factor

        # 修正2：稳定数值计算（可选）
        attn = F.softmax(attn_logits, dim = -1)  # [B, C, C]

        weighted = torch.bmm(attn, V).view(B, C, H, W)
        return f_a + weighted  # 残差连接


class SpatialInteractionV1(nn.Module):
    """优化点：引入位置编码与多头机制"""

    def __init__(self, out_c, num_heads = 8):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(1, out_c, 32, 32))  # 可学习位置编码
        self.flash_attn = FlashCrossAttention(out_c, num_heads)

    def forward(self, f_a, f_b):
        B, C, H, W = f_a.shape

        # 位置编码插值适配
        pos = F.interpolate(self.pos_enc, size = (H, W), mode = 'bilinear')
        f_a = f_a + pos

        # 空间注意力计算
        attn_out = self.flash_attn(
            f_a.flatten(2).permute(0, 2, 1),
            f_b.flatten(2).permute(0, 2, 1),
            f_b.flatten(2).permute(0, 2, 1)
        ).permute(0, 2, 1).view(B, C, H, W)

        return f_a + attn_out  # 残差连接


class SpatialInteractionV2(nn.Module):
    """优化点：引入位置编码与多头机制"""

    def __init__(self, out_c, num_heads = 8):
        super().__init__()
        # 独立初始化h/w编码
        self.pos_enc_h = nn.Parameter(torch.Tensor(1, out_c, 32, 1))  # [1,C,H_base,1]
        self.pos_enc_w = nn.Parameter(torch.Tensor(1, out_c, 1, 32))  # [1,C,1,W_base]

        # 参数初始化（控制幅度）
        nn.init.trunc_normal_(self.pos_enc_h, std = 0.02)
        nn.init.trunc_normal_(self.pos_enc_w, std = 0.02)
        self.flash_attn = FlashCrossAttention(out_c, num_heads)

    def forward(self, f_a, f_b):
        B, C, H, W = f_a.shape

        # 独立插值h/w编码
        pos_h = F.interpolate(
            self.pos_enc_h,
            size = (H, 1),
            mode = 'bilinear',
            align_corners = False
        )  # [1,C,H,1]

        pos_w = F.interpolate(
            self.pos_enc_w,
            size = (1, W),
            mode = 'bilinear',
            align_corners = False
        )  # [1,C,1,W]

        # 广播相加生成完整位置编码
        pos = pos_h + pos_w  # [1,C,H,W]
        # 位置编码插值适配
        f_a = f_a + pos

        # 空间注意力计算
        attn_out = self.flash_attn(
            f_a.flatten(2).permute(0, 2, 1),
            f_b.flatten(2).permute(0, 2, 1),
            f_b.flatten(2).permute(0, 2, 1)
        ).permute(0, 2, 1).view(B, C, H, W)

        return f_a + attn_out  # 残差连接


class SpatialInteractionV3(nn.Module):
    def __init__(self, out_c, num_heads = 8, groups = 8):
        super().__init__()
        # 共享基础位置编码
        self.pos_base = nn.Parameter(torch.Tensor(1, out_c, 32, 32))
        self.proj_h = nn.Conv2d(out_c, out_c, 1)  # 高度方向投影
        self.proj_w = nn.Conv2d(out_c, out_c, 1)  # 宽度方向投影
        nn.init.trunc_normal_(self.pos_base, std = 0.02)
        self.v_proj = nn.Sequential(
            nn.Conv2d(2 * out_c, out_c, 1, groups = groups),
            nn.GELU()
        )
        self.flash_attn = FlashCrossAttention(out_c, num_heads)

    def forward(self, f_a, f_b):
        B, C, H, W = f_a.shape

        # 动态生成方向敏感编码
        pos = F.interpolate(self.pos_base, (H, W), mode = 'bilinear')
        pos_h = self.proj_h(pos.mean(dim = -1, keepdim = True))  # 高度编码 [1,C,H,1]
        pos_w = self.proj_w(pos.mean(dim = -2, keepdim = True))  # 宽度编码 [1,C,1,W]
        pos = pos_h + pos_w

        # 双向位置注入
        f_a = f_a + pos
        f_b = f_b + pos

        V = self.v_proj(torch.cat([f_a, f_b], dim = 1))

        # 注意力计算
        attn_out = self.flash_attn(
            f_a.flatten(2).permute(0, 2, 1).contiguous(),
            f_b.flatten(2).permute(0, 2, 1).contiguous(),
            V.flatten(2).permute(0, 2, 1).contiguous()
        ).permute(0, 2, 1).contiguous().view(B, C, H, W)

        return f_a + attn_out


class SpatialInteractionV3_Directiona(nn.Module):
    def __init__(self, out_c, num_heads = 8, base_size = 32):
        super().__init__()
        # 共享基础位置编码 + 方向投影
        self.pos_base = nn.Parameter(torch.Tensor(1, out_c // 2, base_size, base_size))
        self.proj_a2b = nn.Conv2d(out_c // 2, out_c, 1)  # A→B方向
        self.proj_b2a = nn.Conv2d(out_c // 2, out_c, 1)  # B→A方向
        nn.init.trunc_normal_(self.pos_base, std = 0.02)

        # 轻量Value生成
        self.v_proj = nn.Sequential(
            nn.Conv2d(2 * out_c, out_c, 1),
            nn.GELU()
        )
        self.flash_attn = FlashCrossAttention(out_c, num_heads)

    def forward(self, f_a, f_b, direction = 'res2vm'):
        B, C, H, W = f_a.shape

        # 动态生成方向敏感位置编码
        pos_base = F.interpolate(self.pos_base, (H, W), mode = 'bilinear')
        if direction == 'res2vm':
            pos = self.proj_a2b(pos_base)  # A→B方向投影
            f_a = f_a + pos  # 仅增强查询侧
        else:
            pos = self.proj_b2a(pos_base)  # B→A方向投影
            f_b = f_a + pos  # 仅增强键值侧

        # 注意力计算
        V = self.v_proj(torch.cat([f_a, f_b], dim = 1))
        attn_out = self.flash_attn(
            f_a.flatten(2).permute(0, 2, 1).contiguous(),
            f_b.flatten(2).permute(0, 2, 1).contiguous(),
            V.flatten(2).permute(0, 2, 1).contiguous()
        ).permute(0, 2, 1).contiguous().view(B, C, H, W)
        # 方向自适应残差
        return f_a + attn_out


class SpatialInteractionV3_Lite(nn.Module):
    def __init__(self, out_c, num_heads = 8, groups = 8, base_size = 32):
        super().__init__()
        self.groups = groups
        self.base_size = base_size

        # 共享基础位置编码（尺寸减半）
        self.pos_base = nn.Parameter(torch.Tensor(1, out_c // 2, base_size, base_size))
        nn.init.trunc_normal_(self.pos_base, std = 0.02)

        # 方向敏感投影（深度可分离卷积）
        self.pos_proj = nn.Sequential(
            nn.Conv2d(out_c // 2, out_c // 2, 3, padding = 1, groups = out_c // 2),  # Depthwise
            nn.Conv2d(out_c // 2, out_c, 1),  # Pointwise
            nn.GELU()
        )

        self.v_proj = nn.Sequential(
            nn.Conv2d(2 * out_c, out_c, 1, groups = groups),
            nn.GELU()
        )

        # 高效多头注意力（头维度共享）
        self.flash_attn = FlashCrossAttention(out_c, num_heads)

    def forward(self, f_a, f_b):
        B, C, H, W = f_a.shape

        # 动态位置编码生成
        pos = F.interpolate(self.pos_base, (H, W), mode = 'bilinear', align_corners = False)
        pos = self.pos_proj(pos)  # [B, C, H, W]

        # 双向位置注入（参数共享）
        f_a = f_a + pos * 0.5  # 缩放因子防止过拟合
        f_b = f_b + pos * 0.5

        # 轻量Value生成
        V = self.v_proj(torch.cat([f_a, f_b], dim = 1))

        # 高效注意力计算
        q = f_a.flatten(2).permute(0, 2, 1)
        k = f_b.flatten(2).permute(0, 2, 1)
        v = V.flatten(2).permute(0, 2, 1)

        attn_out = self.flash_attn(q, k, v)
        attn_out = attn_out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # 通道注意力门控残差
        return f_a + attn_out  # 门控残差连接


class SpatialInteractionV4(nn.Module):
    """方向敏感的空间交互模块（需成对使用）"""

    def __init__(self, out_c, num_heads = 8, groups = 8):
        super().__init__()
        assert out_c % groups == 0, f"out_c ({out_c})必须能被groups ({groups})整除"

        # 方向相关的位置编码（A→B和B→A独立）
        self.pos_enc_h_a2b = nn.Parameter(torch.Tensor(1, out_c, 32, 1))  # A→B方向
        self.pos_enc_w_a2b = nn.Parameter(torch.Tensor(1, out_c, 1, 32))
        self.pos_enc_h_b2a = nn.Parameter(torch.Tensor(1, out_c, 32, 1))  # B→A方向
        self.pos_enc_w_b2a = nn.Parameter(torch.Tensor(1, out_c, 1, 32))

        self.v_proj = nn.Sequential(
            nn.Conv2d(2 * out_c, out_c, 1, groups = groups),
            nn.GELU()
        )

        # 参数初始化
        for param in [self.pos_enc_h_a2b, self.pos_enc_w_a2b, self.pos_enc_h_b2a, self.pos_enc_w_b2a]:
            nn.init.trunc_normal_(param, std = 0.02)
        self.flash_attn = FlashCrossAttention(out_c, num_heads)

    def forward(self, f_a, f_b, direction):
        B, C, H, W = f_a.shape

        # 动态选择位置编码参数
        if direction == 'res2vm':
            pos_enc_h = self.pos_enc_h_a2b
            pos_enc_w = self.pos_enc_w_a2b
        else:
            pos_enc_h = self.pos_enc_h_b2a
            pos_enc_w = self.pos_enc_w_b2a

        # 插值生成位置编码
        pos_h = F.interpolate(pos_enc_h, (H, 1), mode = 'bilinear')
        pos_w = F.interpolate(pos_enc_w, (1, W), mode = 'bilinear')
        pos = pos_h + pos_w

        # 方向敏感的特征投影
        if direction == 'res2vm':
            Q = f_a + pos  # Q: ResNet特征 + 位置编码
            K = f_b + pos  # K: VMamba特征 + 同方向编码
            V = self.v_proj(torch.cat([f_a, f_b], dim = 1))
        else:
            Q = f_b + pos  # Q: VMamba特征 + 位置编码
            K = f_a + pos  # K: ResNet特征 + 同方向编码
            V = self.v_proj(torch.cat([f_a, f_b], dim = 1))

        # 注意力计算
        attn_out = self.flash_attn(
            Q.flatten(2).permute(0, 2, 1).contiguous(),
            K.flatten(2).permute(0, 2, 1).contiguous(),
            V.flatten(2).permute(0, 2, 1).contiguous()
        ).permute(0, 2, 1).contiguous().view(B, C, H, W)

        # 残差连接（方向敏感）
        return f_b + attn_out


class SpatialInteractionV5(nn.Module):
    """方向敏感的空间交互模块（需成对使用）"""

    def __init__(self, out_c, num_heads = 8, groups = 8):
        super().__init__()
        assert out_c % groups == 0, f"out_c ({out_c})必须能被groups ({groups})整除"

        # 方向相关的位置编码（A→B和B→A独立）
        self.pos_enc_h_a2b = nn.Parameter(torch.Tensor(1, out_c, 32, 1))  # A→B方向
        self.pos_enc_w_a2b = nn.Parameter(torch.Tensor(1, out_c, 1, 32))
        self.pos_enc_h_b2a = nn.Parameter(torch.Tensor(1, out_c, 32, 1))  # B→A方向
        self.pos_enc_w_b2a = nn.Parameter(torch.Tensor(1, out_c, 1, 32))

        self.v_proj = nn.Sequential(
            nn.Conv2d(2 * out_c, out_c, 1, groups = groups),
            nn.GELU()
        )

        # 参数初始化
        for param in [self.pos_enc_h_a2b, self.pos_enc_w_a2b, self.pos_enc_h_b2a, self.pos_enc_w_b2a]:
            nn.init.trunc_normal_(param, std = 0.02)
        self.flash_attn = FlashCrossAttention(out_c, num_heads)

    def forward(self, f_a, f_b, direction):
        B, C, H, W = f_a.shape

        # 动态选择位置编码参数
        if direction == 'res2vm':
            pos_enc_h = self.pos_enc_h_a2b
            pos_enc_w = self.pos_enc_w_a2b
        else:
            pos_enc_h = self.pos_enc_h_b2a
            pos_enc_w = self.pos_enc_w_b2a

        # 插值生成位置编码
        pos_h = F.interpolate(pos_enc_h, (H, 1), mode = 'bilinear')
        pos_w = F.interpolate(pos_enc_w, (1, W), mode = 'bilinear')
        pos = pos_h + pos_w

        # 方向敏感的特征投影
        if direction == 'res2vm':
            Q = f_a + pos  # Q: ResNet特征 + 位置编码
            K = f_b + pos  # K: VMamba特征 + 同方向编码
            V = self.v_proj(torch.cat([f_a, f_b], dim = 1))
        else:
            Q = f_b + pos  # Q: VMamba特征 + 位置编码
            K = f_a + pos  # K: ResNet特征 + 同方向编码
            V = self.v_proj(torch.cat([f_a, f_b], dim = 1))

        # 注意力计算
        attn_out = self.flash_attn(
            Q.flatten(2).permute(0, 2, 1).contiguous(),
            K.flatten(2).permute(0, 2, 1).contiguous(),
            V.flatten(2).permute(0, 2, 1).contiguous()
        ).permute(0, 2, 1).contiguous().view(B, C, H, W)

        # 残差连接（方向敏感）
        return f_a + attn_out


class FeatureAlignerv2(nn.Module):
    def __init__(self, c_res, c_vm, out_c, num_heads = 8):
        """
        :param c_res:   ResNeXt特征通道数
        :param c_vm:    VMamba特征通道数
        :param out_c:   输出通道数
        :param num_heads: 多头注意力头数
        """
        super().__init__()
        assert out_c % num_heads == 0, "out_c必须能被num_heads整除"

        # ----------------- 通道对齐 -----------------
        self.conv_res = nn.Sequential(
            nn.Conv2d(c_res, out_c, 1),
            nn.GroupNorm(8, out_c)
        ) if c_res != out_c else nn.Identity()

        self.conv_vm = nn.Sequential(
            nn.Conv2d(c_vm, out_c, 1),
            nn.GroupNorm(8, out_c)
        ) if c_vm != out_c else nn.Identity()

        # ----------------- 混合注意力模块 -----------------
        self.channel_interact = ChannelInteractionV3(out_c)
        # self.spatial_interact = SpatialInteractionV4(out_c)  # A→B方向

        self.interact_res2vm = SpatialInteractionV3(out_c)
        self.interact_vm2res = SpatialInteractionV3(out_c)

        # ----------------- 多级融合 -----------------
        # self.fusion = nn.Sequential(
        #     nn.Conv2d(3 * out_c, out_c, 3, padding = 1),
        #     nn.ReLU(inplace = True),
        #     nn.Conv2d(out_c, out_c, 1)
        # )
        self.fusion = nn.Sequential(
            # 深度可分离卷积替代普通3x3卷积
            nn.Conv2d(3 * out_c, 3 * out_c, 3, padding = 1, groups = 3 * out_c),  # 深度卷积
            nn.Conv2d(3 * out_c, out_c, 1),  # 逐点卷积
            nn.ReLU(inplace = True),
            nn.Conv2d(out_c, out_c, 1)
        )

    def forward(self, f_res, f_vm):
        # Step 1: 通道对齐与标准化
        f_res = self.conv_res(f_res)  # [B, C, H, W]
        f_vm = self.conv_vm(f_vm)  # [B, C, H, W]

        # Step 2: 双向混合注意力
        # 通道级交互
        f_vm_ch = self.channel_interact(f_res, f_vm, 'res2vm')  # ResNeXt→VMamba
        f_res_ch = self.channel_interact(f_vm, f_res, 'vm2res')  # VMamba→ResNeXt

        # 空间级交互
        f_res_sp = self.spatial_interact(f_res, f_vm, 'res2vm')
        f_vm_sp = self.spatial_interact(f_vm, f_res, 'vm2res')
        # f_res_sp = self.interact_res2vm(f_res, f_vm)
        # f_vm_sp = self.interact_vm2res(f_vm, f_res)

        # Step 3: 多级特征融合
        fused = torch.cat([f_res_ch, f_vm_ch, f_res_sp + f_vm_sp], dim = 1)

        return f_res, f_vm, self.fusion(fused)


class FeatureAlignerv3(nn.Module):
    def __init__(self, c_res, c_vm, out_c, num_heads = 8):
        """
        :param c_res:   ResNeXt特征通道数
        :param c_vm:    VMamba特征通道数
        :param out_c:   输出通道数
        :param num_heads: 多头注意力头数
        """
        super().__init__()
        assert out_c % num_heads == 0, "out_c必须能被num_heads整除"

        # ----------------- 通道对齐 -----------------
        self.conv_res = nn.Sequential(
            nn.Conv2d(c_res, out_c, 1),
            nn.GroupNorm(8, out_c)
        ) if c_res != out_c else nn.Identity()

        self.conv_vm = nn.Sequential(
            nn.Conv2d(c_vm, out_c, 1),
            nn.GroupNorm(8, out_c)
        ) if c_vm != out_c else nn.Identity()

        # ----------------- 混合注意力模块 -----------------
        self.channel_interact = ChannelInteractionV3(out_c)
        # self.spatial_interact = SpatialInteractionV3_Directiona(out_c)  # A→B方向

        self.interact_res2vm = SpatialInteractionV3(out_c)
        self.interact_vm2res = SpatialInteractionV3(out_c)

        # ----------------- 多级融合 -----------------
        # self.fusion = nn.Sequential(
        #     nn.Conv2d(3 * out_c, out_c, 3, padding = 1),
        #     nn.ReLU(inplace = True),
        #     nn.Conv2d(out_c, out_c, 1)
        # )
        self.fusion = nn.Sequential(
            # 深度可分离卷积替代普通3x3卷积
            nn.Conv2d(3 * out_c, 3 * out_c, 3, padding = 1, groups = 3 * out_c),  # 深度卷积
            nn.Conv2d(3 * out_c, out_c, 1),  # 逐点卷积
            nn.ReLU(inplace = True),
            nn.Conv2d(out_c, out_c, 1)
        )

    def forward(self, f_res, f_vm):
        # Step 1: 通道对齐与标准化
        f_res = self.conv_res(f_res)  # [B, C, H, W]
        f_vm = self.conv_vm(f_vm)  # [B, C, H, W]

        # Step 2: 双向混合注意力
        # 通道级交互
        f_vm_ch = self.channel_interact(f_res, f_vm, 'res2vm')  # ResNeXt→VMamba
        f_res_ch = self.channel_interact(f_vm, f_res, 'vm2res')  # VMamba→ResNeXt

        # 空间级交互
        # f_res_sp = self.spatial_interact(f_res, f_vm, 'res2vm')
        # f_vm_sp = self.spatial_interact(f_vm, f_res, 'vm2res')
        f_res_sp = self.interact_res2vm(f_res, f_vm)
        f_vm_sp = self.interact_vm2res(f_vm, f_res)

        # Step 3: 多级特征融合
        fused = torch.cat([f_res_ch, f_vm_ch, f_res_sp + f_vm_sp], dim = 1)

        return f_res, f_vm, self.fusion(fused)


# 互补特征提取模块
# 通过残差掩码学习各分支独有的互补特征。
class ComplementaryExtractorV3(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        # 坐标注意力（空间定位）
        self.coord_attn = CoordAtt(in_c, in_c)
        # 通道注意力（跨模态交互）
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_c, in_c // 16, 1, bias = False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace = True),
            nn.Conv2d(in_c // 16, in_c, 1, bias = False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))

        # 第一阶段：通道选择
        channel_weights = self.sigmoid(max_out + avg_out)  # [B,C,1,1]
        x_channel = x * channel_weights  # 增强重要通道

        # 第二阶段：空间调制
        aw, ah = self.coord_attn(x_channel)  # 基于通道增强后的特征
        return x_channel * aw * ah  # 分离式加权


# 动态门控注入模块
# 根据公共特征动态调节互补特征的注入强度。
class DynamicGateV4(nn.Module):
    def __init__(self, c_common, c_comp):
        super().__init__()
        # 三阶轻量门控
        self.gate_net = nn.Sequential(
            nn.Conv2d(c_common + c_comp, c_comp // 4, 1),  # 输入拼接公共特征和独有特征
            nn.BatchNorm2d(c_comp // 4),
            nn.ReLU(),
            nn.Conv2d(c_comp // 4, c_comp, 3, padding = 1, groups = 8),  # 分组卷积提升效率
            nn.Sigmoid()
        )
        # 残差系数（可学习）
        self.res_coef = nn.Parameter(torch.tensor(0.2))

    def forward(self, f_common, delta):
        # 拼接公共特征与独有特征
        gate_input = torch.cat([
            F.adaptive_avg_pool2d(f_common, delta.shape[2:]),  # 尺寸对齐
            delta
        ], dim = 1)

        gate = self.gate_net(gate_input)
        return delta * gate + self.res_coef * delta  # 残差连接保留部分原始信息


# 最新版Dual Complementary Fusion (DCF)
class MSSF(nn.Module):
    def __init__(self, c_res, c_vm, fused_c = 256):
        super().__init__()
        # Step 1: 特征对齐与公共特征提取
        self.aligner = FeatureAlignerv3(c_res, c_vm, fused_c)

        # Step 2: 互补特征提取
        self.delta_res = ComplementaryExtractorV3(fused_c)
        self.delta_vm = ComplementaryExtractorV3(fused_c)
        # 其中α为可学习参数（初始化为0.7）
        init_s = 0.7
        self.alpha = torch.nn.Parameter(torch.tensor(math.log(init_s / (1.0 - init_s + 1e-6)), dtype = torch.float32))

        # Step 3: 动态门控
        self.gate_res = DynamicGateV4(fused_c, fused_c)
        self.gate_vm = DynamicGateV4(fused_c, fused_c)

    def forward(self, f_res, f_vm):
        # Step 1: 公共特征
        f_res, f_vm, f_common = self.aligner(f_res, f_vm)

        # 部分梯度回传
        # α为可学习参数或固定值
        s = torch.sigmoid(self.alpha)  # in (0,1)
        s = s.clamp(0.05, 0.95)  # 防止过极端
        # f_common_detached = f_common * (1.0 - s)
        # # f_common_detached = f_common * (1.0 - s) + f_common.detach() * s

        # s = 0.05 + 0.90 * torch.sigmoid(self.alpha)
        f_common_detached = (
            f_common * (1.0 - s)
            + f_common.detach() * s
            + (s - s.detach()) * f_common.detach()
        )
        # f_common_detached = f_common * (1.0 - s) + f_common.detach() * s
        # f_common_detached = f_common * (1.0 - s)

        # 计算差异时直接取绝对值
        diff_res = f_res - f_common_detached
        diff_vm = f_vm - f_common_detached

        # 平滑绝对值，避免在 0 点的子梯度问题
        eps = 1e-6
        delta_res = self.delta_res(torch.sqrt(diff_res * diff_res + eps))
        delta_vm = self.delta_vm(torch.sqrt(diff_vm * diff_vm + eps))

        # Step 3: 动态注入
        # gate_res = self.gate_res(f_common, delta_res)
        # gate_vm = self.gate_vm(f_common, delta_vm)
        gate_res = self.gate_res(f_common_detached, delta_res)
        gate_vm = self.gate_vm(f_common_detached, delta_vm)
        fused = f_common + gate_res + gate_vm

        return fused

# 定义一个简单的双层卷积块作为 Baseline
class SimpleConvBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Sequential(
            # 第一个 3x3 卷积
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            # 第二个 3x3 卷积
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class simple_MSSF(nn.Module):
    def __init__(self, c_res, c_vm, fused_c=256):
        super().__init__()
        # Step 1: 特征对齐与公共特征提取 (保持不变)
        self.aligner = FeatureAlignerv3(c_res, c_vm, fused_c)

        # Step 2: 互补特征提取 -> 【替换为简单的卷积块】
        # 原来是: self.delta_res = ComplementaryExtractorV3(fused_c)
        # 现在替换为:
        self.delta_res = SimpleConvBlock(fused_c)
        self.delta_vm = SimpleConvBlock(fused_c)
        
        # -----------------------------------------------------------
        # 注意：为了控制变量，如果你想证明仅仅是“结构”的区别，
        # 建议保留原本的梯度阻尼逻辑 (alpha)。
        # 这样对比的就是：[Diff -> Attention] vs [Diff -> Conv]
        # -----------------------------------------------------------
        init_s = 0.7
        self.alpha = torch.nn.Parameter(torch.tensor(math.log(init_s / (1.0 - init_s + 1e-6)), dtype=torch.float32))

    def forward(self, f_res, f_vm):
        # Step 1: 公共特征
        f_res, f_vm, f_common = self.aligner(f_res, f_vm)

        # -----------------------------------------------------------
        # 关键提示：做对比实验时，这里的逻辑必须和你【主实验】保持一致！
        # 如果你主实验用的是旧逻辑 (detach)，这里也请务必改回 detach 逻辑。
        # 下面是你提供的代码逻辑 (无 detach)：
        # -----------------------------------------------------------
        
        # 计算阻尼系数
        s = torch.sigmoid(self.alpha)
        s = s.clamp(0.05, 0.95)
        
        # 应用梯度/信息阻尼
        f_common_detached = f_common * (1.0 - s)
        # f_common_detached = f_common * (1.0 - s) + f_common.detach() * s

        # 计算差异 (Difference)
        diff_res = f_res - f_common_detached
        diff_vm = f_vm - f_common_detached

        # 平滑绝对值
        eps = 1e-6
        diff_res_abs = torch.sqrt(diff_res * diff_res + eps)
        diff_vm_abs = torch.sqrt(diff_vm * diff_vm + eps)

        # Step 2: 简单的互补特征提取
        # 输入绝对差异图，通过简单的卷积层提取特征
        delta_res = self.delta_res(diff_res_abs)
        delta_vm = self.delta_vm(diff_vm_abs)

        fused = f_common + delta_res + delta_vm

        return fused
