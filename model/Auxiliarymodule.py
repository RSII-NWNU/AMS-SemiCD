import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------- 辅助模块 ----------------------
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(self.avg_pool(x))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding = 1)

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        return x * torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim = 1)))


# CBAM卷积注意力模块
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction = 16, spatial_kernel = 7):
        super().__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias = False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace = True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias = False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size = spatial_kernel,
                              padding = spatial_kernel // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim = 1)))
        x = spatial_out * x
        return x


class FlashCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        B, L, _ = query.shape

        # 投影到Q/K/V
        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 使用PyTorch内置Flash Attention（v2优化）
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = None,
            dropout_p = 0.0,
            is_causal = False
        )

        # 恢复形状并输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(attn_output)


class CubicCoordAtt(nn.Module):
    def __init__(self, dim, group, kernel):
        super().__init__()
        self.H_spatial_att = SpatialStripAtt(dim, group = group, kernel = kernel, H = True)
        self.W_spatial_att = SpatialStripAtt(dim, group = group, kernel = kernel, H = False)
        self.coord_att = CoordAtt(inp = dim, oup = dim)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # 空间条带注意力
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)

        # 坐标注意力
        a_w, a_h = self.coord_att(x)

        # 维度自适应融合
        return self.gamma * (out * a_w * a_h) + self.beta * x


class SpatialStripAtt(nn.Module):
    def __init__(self, dim, kernel = 5, group = 2, H = True):
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.group = group

        # 反射填充设置
        self.padding = (pad, pad, 0, 0) if H else (0, 0, pad, pad)
        self.pad = nn.ReflectionPad2d(self.padding)

        # 动态核生成
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size = 1, bias = False)
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        n, c, h, w = x.shape

        # 生成动态卷积核
        filter = self.conv(self.ap(x))  # [n, group*k, 1, 1]
        filter = filter.view(n, self.group, self.k, -1).unsqueeze(2)  # [n, g, 1, k, L]

        # 展开特征图
        x_unfold = F.unfold(self.pad(x), kernel_size = self.kernel)  # [n, c*k, L]
        x_unfold = x_unfold.view(n, self.group, c // self.group, self.k, -1)  # [n, g, c/g, k, L]

        # 动态卷积操作
        out = torch.sum(x_unfold * self.act(filter), dim = 3)  # [n, g, c/g, L]
        return out.view(n, c, h, w)


# class MultiScaleSpatialAtt(nn.Module):
#     def __init__(self, dim, scales = [3, 5, 7], groups = 4):
#         super().__init__()
#         self.scales = sorted(scales)
#         self.groups = groups
#         self.dim = dim

#         # 水平与垂直分支初始化
#         self.h_conv_layers = nn.ModuleList()
#         self.w_conv_layers = nn.ModuleList()
#         self.h_pads = nn.ModuleList()
#         self.w_pads = nn.ModuleList()

#         for k in self.scales:
#             pad = k // 2
#             # 水平分支
#             self.h_pads.append(nn.ReflectionPad2d((pad, pad, 0, 0)))
#             self.h_conv_layers.append(nn.Conv2d(dim, dim * k, 1))  # 关键修复：输出通道为dim*k

#             # 垂直分支
#             self.w_pads.append(nn.ReflectionPad2d((0, 0, pad, pad)))
#             self.w_conv_layers.append(nn.Conv2d(dim, dim * k, 1))  # 关键修复：输出通道为dim*k

#         # 多尺度融合
#         self.fusion = nn.Conv2d(len(scales) * dim, dim, 3, padding = 1)

#         # 尺度注意力
#         self.scale_att = nn.Sequential(
#             nn.Conv2d(dim, len(scales), 3, padding = 1),
#             nn.Softmax(dim = 1)
#         )

#         # 残差参数
#         self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
#         self.beta = nn.Parameter(torch.ones(1, dim, 1, 1))

#         # 跨时相核生成器
#         self.kernel_fuser = nn.Sequential(
#             nn.Conv2d(2 * dim, dim, 3, padding = 1),
#             nn.GroupNorm(num_groups = 8, num_channels = dim),
#             nn.GELU()
#         )

#     def forward(self, x1, x2_aligned):
#         x = self.kernel_fuser(torch.cat([x1, x2_aligned], dim = 1))
#         B, C, H, W = x.shape
#         outputs = []

#         for i, k in enumerate(self.scales):
#             # --- 水平处理 ---
#             # 生成动态核 [B, groups, C//groups, k, H, W]
#             kernel_h = self.h_conv_layers[i](x).contiguous().view(B, self.groups, C // self.groups, k, H, W).sigmoid()

#             # 展开输入 [B, groups, C//groups, k, H, W]
#             padded_h = self.h_pads[i](x)
#             unfolded_h = F.unfold(padded_h, (1, k)).contiguous()  # [B, C*k, L]
#             unfolded_h = unfolded_h.view(B, self.groups, C // self.groups, k, H, W)

#             # 动态卷积与聚合
#             att_h = (unfolded_h * kernel_h).sum(dim = 3).view(B, C, H, W)

#             # --- 垂直处理 ---
#             # 生成动态核 [B, groups, C//groups, k, H, W]
#             kernel_w = self.w_conv_layers[i](x).contiguous().view(B, self.groups, C // self.groups, k, H, W).sigmoid()

#             # 展开输入 [B, groups, C//groups, k, H, W]
#             padded_w = self.w_pads[i](x)
#             unfolded_w = F.unfold(padded_w, (k, 1)).contiguous()  # [B, C*k, L]
#             unfolded_w = unfolded_w.view(B, self.groups, C // self.groups, k, H, W)

#             # 动态卷积与聚合
#             att_w = (unfolded_w * kernel_w).sum(dim = 3).view(B, C, H, W)

#             outputs.append(att_h + att_w)

#         # 多尺度融合
#         fused = self.fusion(torch.cat(outputs, dim = 1))

#         # 尺度注意力加权
#         scale_weights = self.scale_att(fused)  # [B, num_scales, H, W]
#         final = sum(weight.unsqueeze(1) * feat for weight, feat in zip(scale_weights.unbind(1), outputs))

#         # 残差连接
#         return self.gamma * final + self.beta * x
        
class MultiScaleSpatialAtt(nn.Module):
    """
    分析与修正后的版本 v2:
    1. 动态核生成采用纯局部信息驱动。
    2. 采用 F.unfold 实现逐像素动态卷积。
    3. 多尺度融合：舍弃 fusion 模块，直接由 scale_att 从高维特征生成权重。
    """
    def __init__(self, dim, scales=[3, 5, 7], groups=8):
        super().__init__()
        self.scales = sorted(scales)
        self.groups = groups
        self.dim = dim

        self.kernel_fuser = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=dim),
            nn.GELU()
        )

        self.kernel_generators = nn.ModuleList()
        self.h_pads = nn.ModuleList()
        self.w_pads = nn.ModuleList()

        for k in self.scales:
            pad = k // 2
            self.h_pads.append(nn.ReflectionPad2d((pad, pad, 0, 0)))
            self.w_pads.append(nn.ReflectionPad2d((0, 0, pad, pad)))
            self.kernel_generators.append(nn.Conv2d(dim, 2 * dim * k, 1))

        num_scales = len(scales)
        
        # 定义尺度注意力模块，使其直接处理拼接后的高维特征
        attention_inter_dim = max(dim // 4, num_scales) 
        self.scale_att = nn.Sequential(
            # 输入通道数从 dim 修改为 num_scales * dim
            nn.Conv2d(in_channels=num_scales * dim, out_channels=attention_inter_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels=attention_inter_dim, out_channels=num_scales, kernel_size=3, padding=1, bias=False),
            nn.Softmax(dim=1)
        )
        # -----------------------------------------------------------------

    def forward(self, x1, x2_aligned):
        x = self.kernel_fuser(torch.cat([x1, x2_aligned], dim=1))
        B, C, H, W = x.shape

        outputs = []
        for i, k in enumerate(self.scales):
            kernel_map = self.kernel_generators[i](x)
            kernel_h_map, kernel_w_map = torch.split(kernel_map, [C * k, C * k], dim=1)
            
            kernel_h = kernel_h_map.contiguous().view(B, self.groups, C // self.groups, k, H, W).sigmoid()
            padded_h = self.h_pads[i](x)
            unfolded_h = F.unfold(padded_h, (1, k)).contiguous().view(B, self.groups, C // self.groups, k, H, W)
            att_h = (unfolded_h * kernel_h).sum(dim=3).view(B, C, H, W)
            
            kernel_w = kernel_w_map.contiguous().view(B, self.groups, C // self.groups, k, H, W).sigmoid()
            padded_w = self.w_pads[i](x)
            unfolded_w = F.unfold(padded_w, (k, 1)).contiguous().view(B, self.groups, C // self.groups, k, H, W)
            att_w = (unfolded_w * kernel_w).sum(dim=3).view(B, C, H, W)
            
            outputs.append(att_h + att_w)

        # 4. 多尺度融合与尺度注意力加权
        # --- 修正点 2: 直接将拼接后的特征送入 scale_att ---
        concatenated_features = torch.cat(outputs, dim=1)
        scale_weights = self.scale_att(concatenated_features)  # [B, num_scales, H, W]
        # --------------------------------------------------
        
        final = sum(weight.unsqueeze(1) * feat for weight, feat in zip(scale_weights.unbind(1), outputs))
        
        return final


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size = 1, stride = 1, padding = 0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim = 2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim = 2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_h = a_h.expand(-1, -1, h, w)
        a_w = a_w.expand(-1, -1, h, w)

        # out = identity * a_w * a_h

        return a_w, a_h


class h_swish(nn.Module):
    def __init__(self, inplace = True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace = inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace = True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace = inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SpectralCoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction = 32, kernels = [3, 7]):
        super().__init__()
        # 多尺度分支
        self.global_att = GlobalFreqCoordAtt(inp, reduction)
        self.local_att = nn.ModuleList([
            LocalFreqCoordAtt(inp, k, reduction) for k in kernels
        ])

        # 跨尺度融合
        self.fusion = nn.Sequential(
            nn.Conv2d(len(kernels) + 1, 4, 3, padding = 1),
            nn.Hardswish(),
            nn.Conv2d(4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 全局分支
        g_low, g_high = self.global_att(x)
        global_out = g_low + 0.6 * g_high  # 经验系数

        # 局部分支
        local_outs = [att(x)[1] for att in self.local_att]  # 仅取高频

        # 特征拼接与融合 [B,1+len(kernels),H,W]
        feat_stack = torch.cat([global_out.unsqueeze(1)] + [lo.unsqueeze(1) for lo in local_outs], dim = 1)
        fused_weights = self.fusion(feat_stack)

        return x * fused_weights.squeeze(1)


class GlobalFreqCoordAtt(nn.Module):
    """ 全局频谱分解的坐标注意力 """

    def __init__(self, inp, reduction):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 低频通路
        mip = max(8, inp // reduction)
        self.conv_low = nn.Sequential(
            nn.Conv2d(inp, mip, 1),
            nn.BatchNorm2d(mip),
            h_swish()
        )

        # 高频通路
        self.high_conv = nn.Conv2d(inp, mip // 2, 3, padding = 1)

        # 动态权重生成
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mip + mip // 2, 2, 1),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        # 低频分量
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        low_feat = torch.cat([x_h, x_w], dim = 2)
        low_feat = self.conv_low(low_feat)

        # 高频分量
        high_feat = self.high_conv(x - F.interpolate(low_feat, x.shape[2:]))

        # 动态融合
        weights = self.gate(torch.cat([low_feat, high_feat], dim = 1))
        return weights[:, 0] * low_feat, weights[:, 1] * high_feat


class LocalFreqCoordAtt(nn.Module):
    """ 局部频谱分解的坐标注意力 """

    def __init__(self, inp, kernel, reduction):
        super().__init__()
        self.pool = nn.AvgPool2d((kernel, 1) if kernel % 2 else (1, kernel), stride = 1)
        self.pad = nn.ReflectionPad2d(kernel // 2)

        # 局部高低频分解
        self.low_conv = nn.Conv2d(inp, inp // reduction, 1)
        self.high_conv = nn.Conv2d(inp, inp // reduction, 3, padding = 1)

        # 坐标注意力
        self.ca = CoordAtt(inp // reduction, inp // reduction)

    def forward(self, x):
        # 局部低频
        x_low = self.pool(self.pad(x))
        low_feat = self.low_conv(x_low)

        # 局部高频
        high_feat = self.high_conv(x - x_low)

        # 坐标注意力调制
        a_w, a_h = self.ca(high_feat)
        return low_feat, high_feat * (a_w + a_h) / 2


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride = 1, padding = 0, dilation = 1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size = kernel_size, stride = stride,
                              padding = padding, dilation = dilation, bias = False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# class AdaPool(nn.Module):
#     def __init__(self, kernel_size, stride = None, padding = 0):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride or kernel_size
#         self.padding = padding
#
#         # 初始化beta为可学习参数
#         self.beta = nn.Parameter(torch.tensor(0.5))
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         beta = self.sigmoid(self.beta).view(1, 1, 1, 1)  # 修正广播维度
#         return AdaPool2d(
#             x, beta = beta,
#             kernel_size = self.kernel_size,
#             stride = self.stride,
#             padding = self.padding
#         )
#
#
# class EDSCWPool(nn.Module):
#     pass
#
#
# class IDWPool(nn.Module):
#     pass

