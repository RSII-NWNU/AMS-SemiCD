import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from .mssf import MSSF
from .msdb import MSDB
from .agar import CBAM, AGAR
from .vmamba.vmamba import Backbone_VSSM

__all__ = ["SemiModel"]


class SemiModel(nn.Module):
    def __init__(self, vmamba_weight_path = None, res_pretrained = False):
        super().__init__()
        # 按照RESNEXT四层进行拆分
        if res_pretrained:
            resnext50 = models.resnext50_32x4d(weights = models.ResNeXt50_32X4D_Weights.DEFAULT)
        else:
            resnext50 = models.resnext50_32x4d()
        self.firstconv = resnext50.conv1
        self.firstbn = resnext50.bn1
        self.firstrelu = resnext50.relu
        self.firstmaxpool = resnext50.maxpool
        self.res1 = resnext50.layer1  # output: B x256x56x56 (1/4尺寸)
        self.res2 = resnext50.layer2  # output: B x512x28x28 (1/8)
        self.res3 = resnext50.layer3  # output: B x1024x14x14 (1/16)
        self.res4 = resnext50.layer4  # output: B x2048x7x7 (1/32)

        # vmamba
        vmamba_small_m2 = Backbone_VSSM(out_indices = (0, 1, 2, 3),
                                        pretrained = vmamba_weight_path,
                                        depths = [2, 2, 15, 2], dims = 96, drop_path_rate = 0.3,
                                        patch_size = 4, in_chans = 3, num_classes = 1000,
                                        ssm_d_state = 1, ssm_ratio = 2.0, ssm_dt_rank = "auto", ssm_act_layer = "silu",
                                        ssm_conv = 3, ssm_conv_bias = False, ssm_drop_rate = 0.0,
                                        ssm_init = "v0", forward_type = "v05_noz",
                                        mlp_ratio = 4.0, mlp_act_layer = "gelu", mlp_drop_rate = 0.0, gmlp = False,
                                        patch_norm = True, norm_layer = "ln2d",
                                        downsample_version = "v3", patchembed_version = "v2",
                                        use_checkpoint = False, posembed = False, imgsize = 256)
        # self.vmamba = vmamba_base_m2  # output: 128 256 512 1024
        self.vmamba = vmamba_small_m2  # output: 96 192 384 768

        # res+vmamba 异构融合
        self.cgf1 = MSSF(256, 96, 128)  # output: B x128x56x56 (1/4尺寸)
        self.cgf2 = MSSF(512, 192, 256)  # output: B x256x28x28 (1/8)
        self.cgf3 = MSSF(1024, 384, 512)  # output: B x512x14x14 (1/16)
        self.cgf4 = MSSF(2048, 768, 1024)  # output: B x1024x7x7 (1/32)

        # 差异增强模块
        self.dbdfe1 = MSDB(128)
        self.dbdfe2 = MSDB(256)
        self.dbdfe3 = MSDB(512)
        self.dbdfe4 = MSDB(1024)

        # todo 测试上采样和转置卷积
        self.upsample2 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.upsample4 = nn.Upsample(scale_factor = 4, mode = 'bilinear', align_corners = True)
        self.upsample8 = nn.Upsample(scale_factor = 8, mode = 'bilinear', align_corners = True)
        # self.up1 = nn.ConvTranspose2d(4 * dim + 512, 2 * dim + 256, kernel_size = 2, stride = 2)
        # self.up2 = nn.ConvTranspose2d(4 * dim + 512, 2 * dim + 256, kernel_size = 2, stride = 2)
        # self.up3 = nn.ConvTranspose2d(4 * dim + 512, 2 * dim + 256, kernel_size = 2, stride = 2)
        # self.up4 = nn.ConvTranspose2d(dim + 128 + 2 * dim + 256, dim, kernel_size = 4, stride = 4)

        # 融合模块
        self.fusion1 = AGAR(256, 128)
        self.fusion2 = AGAR(512, 256)
        self.fusion3 = AGAR(1024, 512)

        self.sam_p4 = CBAM(1024)
        self.sam_p3 = CBAM(512)
        self.sam_p2 = CBAM(256)
        self.sam_p1 = CBAM(128)

        self.conv1 = nn.Conv2d(in_channels = 1024, out_channels = 96, kernel_size = 1, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 512, out_channels = 96, kernel_size = 1, stride = 1)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 96, kernel_size = 1, stride = 1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 96, kernel_size = 1, stride = 1)

        self.agant1 = self._make_agant_layer(32 * 3, 32 * 2)
        self.agant2 = self._make_agant_layer(32 * 2, 32)
        self.out_conv = nn.Conv2d(32 * 1, 1, kernel_size = 1, stride = 1, bias = True)

    def forward(self, A, B):
        # 1. 拼接输入：将 A 和 B 在 batch 维度拼接 (Batch -> 2*Batch)
        # 假设 A, B shape 为 [B, 3, H, W], X 变为 [2B, 3, H, W]
        x = torch.cat([A, B], dim=0)

        # ---------------- ResNeXt 分支 (一次前向) ----------------
        # 注意：这里需要把原来分步写的 res1, res2... 改为对 x 处理
        # 如果 self.res1 等层包含 BN，这样做可以利用更大的 Batch 统计信息
        
        firstconv = self.firstconv(x)
        firstbn = self.firstbn(firstconv)
        firstrelu = self.firstrelu(firstbn)
        firstmaxpool = self.firstmaxpool(firstrelu)

        res_layer1 = self.res1(firstmaxpool)
        res_layer2 = self.res2(res_layer1)
        res_layer3 = self.res3(res_layer2)
        res_layer4 = self.res4(res_layer3)

        # ---------------- VMamba 分支 (一次前向) ----------------
        # vmamba 的输出通常是一个列表或元组
        vmamba_outs = self.vmamba(x)
        # 假设 vmamba 输出是 tuple: (v_l1, v_l2, v_l3, v_l4)
        # 每个特征图的 shape 也是 [2B, C, H, W]
        vmamba_layer1, vmamba_layer2, vmamba_layer3, vmamba_layer4 = vmamba_outs

        # ---------------- 特征拆分 (Split) ----------------
        # 在进入 CGF (Cross-modal Gated Fusion) 之前，必须把 A 和 B 拆开
        # 因为 CGF 里面是 A 和 A 融合，B 和 B 融合 (或者是 A/B 交互，视具体逻辑而定)
        # 但通常后续的 Difference 模块需要区分 A 和 B
        
        # 辅助函数：将张量切分为 A 和 B 两部分
        def split_feat(feat):
            # chunk(2, dim=0) 将 2B 切成两个 B
            return torch.chunk(feat, 2, dim=0)

        # 拆分 ResNeXt 特征
        res_layer1_A, res_layer1_B = split_feat(res_layer1)
        res_layer2_A, res_layer2_B = split_feat(res_layer2)
        res_layer3_A, res_layer3_B = split_feat(res_layer3)
        res_layer4_A, res_layer4_B = split_feat(res_layer4)

        # 拆分 VMamba 特征
        vmamba_layer1_A, vmamba_layer1_B = split_feat(vmamba_layer1)
        vmamba_layer2_A, vmamba_layer2_B = split_feat(vmamba_layer2)
        vmamba_layer3_A, vmamba_layer3_B = split_feat(vmamba_layer3)
        vmamba_layer4_A, vmamba_layer4_B = split_feat(vmamba_layer4)

        # 跨模态门控融合
        cgf1_A = self.cgf1(res_layer1_A, vmamba_layer1_A)
        cgf2_A = self.cgf2(res_layer2_A, vmamba_layer2_A)
        cgf3_A = self.cgf3(res_layer3_A, vmamba_layer3_A)
        cgf4_A = self.cgf4(res_layer4_A, vmamba_layer4_A)
        cgf1_B = self.cgf1(res_layer1_B, vmamba_layer1_B)
        cgf2_B = self.cgf2(res_layer2_B, vmamba_layer2_B)
        cgf3_B = self.cgf3(res_layer3_B, vmamba_layer3_B)
        cgf4_B = self.cgf4(res_layer4_B, vmamba_layer4_B)

        # 多分支差异特征增强模块
        dbdfe1 = self.dbdfe1(cgf1_A, cgf1_B)
        dbdfe2 = self.dbdfe2(cgf2_A, cgf2_B)
        dbdfe3 = self.dbdfe3(cgf3_A, cgf3_B)
        dbdfe4 = self.dbdfe4(cgf4_A, cgf4_B)
        # dbdfe1 = torch.add(cgf1_A, cgf1_B)
        # dbdfe2 = torch.add(cgf2_A, cgf2_B)
        # dbdfe3 = torch.add(cgf3_A, cgf3_B)
        # dbdfe4 = torch.add(cgf4_A, cgf4_B)

        # 多尺度融合
        dbdfe4 = self.sam_p4(dbdfe4)
        fusion3 = self.fusion3(dbdfe3, dbdfe4)  # 1024
        fusion3 = self.sam_p3(fusion3)
        fusion2 = self.fusion2(dbdfe2, fusion3)  # 512
        fusion2 = self.sam_p2(fusion2)
        fusion1 = self.fusion1(dbdfe1, fusion2)  # 256
        fusion1 = self.sam_p1(fusion1)

        # 上采样
        up8 = self.upsample8(dbdfe4)
        up4 = self.upsample4(fusion3)
        up2 = self.upsample2(fusion2)

        # 降维
        up8 = self.conv1(up8)
        up4 = self.conv2(up4)
        up2 = self.conv3(up2)
        up1 = self.conv4(fusion1)

        y = up1 + up2 + up4 + up8

        y1 = self.agant1(y)
        y2 = self.agant2(y1)
        y3 = self.out_conv(y2)

        # 取张量A的最后两个维度的尺寸，作为插值后的目标尺寸
        return (F.interpolate(y, size = A.size()[2:], mode = 'bilinear',
                              align_corners = True),
                F.interpolate(y3, size = A.size()[2:], mode = 'bilinear',
                              align_corners = True))  # , diff_combined

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size = 1,
                      stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace = True)
        )
        return layers
