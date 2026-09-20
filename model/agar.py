import torch
from einops import rearrange
from torch import nn

from .Auxiliarymodule import ChannelAttention, CBAMLayer, BasicConv2d


# 层间融合
class AGAR(nn.Module):
    '''
    最新版本的ACFF 4.21,将cat改成+，去掉卷积
    '''

    def __init__(self, in_c, out_c):
        super().__init__()

        self.base_conv = BasicConv2d(in_c, out_c, 1, 1, 0)

        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)

        # self.ca = ChannelAttention(out_c, 16)
        self.frfn = FRFN(out_c, out_c * 4)

    def forward(self, f_low, f_high):
        # _,c,h,w = f_low.shape
        # f4上采样，通道数变成原来的1/2,长宽变为原来的2倍
        f_high = self.base_conv(self.up(f_high))

        f_cat = f_high + f_low

        adaptive_w = self.frfn(f_cat)

        out = f_low * adaptive_w + f_high * (1 - adaptive_w)  # B,C_l,h,w
        return out


class FRFN(nn.Module):
    def __init__(self, dim = 32, hidden_dim = 128, split_ratio = 0.5, act_layer = nn.GELU):
        super().__init__()
        self.dim_conv = int(dim * split_ratio)
        self.dim_untouched = dim - self.dim_conv

        # 局部卷积分支
        self.partial_conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias = False)

        # 通道门控机制
        self.linear1 = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            act_layer()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups = hidden_dim, kernel_size = 3, padding = 1),
            act_layer()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid()  # 显式生成权重
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim = 1)
        x1 = self.partial_conv(x1)
        x_processed = torch.cat([x1, x2], dim = 1)

        x_reshaped = rearrange(x_processed, 'b c h w -> b (h w) c')
        gate, residual = self.linear1(x_reshaped).chunk(2, dim = -1)

        gate = rearrange(gate, 'b (h w) c -> b c h w', h = H, w = W)
        gate = self.dwconv(gate)
        gate = rearrange(gate, 'b c h w -> b (h w) c', h = H, w = W)

        fused = gate * residual
        output = self.linear2(fused)
        output = rearrange(output, 'b (h w) c -> b c h w', h = H, w = W)
        # 调整形状以适配 LayerNorm
        output = rearrange(output, 'b c h w -> b (h w) c')
        output = self.norm(output)
        output = rearrange(output, 'b (h w) c -> b c h w', h = H, w = W)

        return x * output


class CBAM(nn.Module):
    def __init__(self, mid_d):
        super().__init__()
        self.mid_d = mid_d

        self.cbam = CBAMLayer(channel = self.mid_d)

        self.base_conv = BasicConv2d(self.mid_d, self.mid_d, 3, 1, 1)

    def forward(self, x):
        context = self.cbam(x)

        x_out = self.base_conv(context)

        return x_out
