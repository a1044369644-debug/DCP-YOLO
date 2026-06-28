import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad

class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class DASI(nn.Module):
    def __init__(self, in_features1,in_features2,in_features3, out_features) -> None:
        super().__init__()
        self.bag = Bag()
        self.tail_conv = nn.Sequential(
            conv_block(in_features=out_features,
                       out_features=out_features,
                       kernel_size=(1, 1),
                       padding=(0, 0),
                       norm_type=None,
                       activation=False)
        )
        self.conv = nn.Sequential(
            conv_block(in_features=out_features // 2,
                       out_features=out_features // 4,
                       kernel_size=(1, 1),
                       padding=(0, 0),
                       norm_type=None,
                       activation=False)
        )
        self.bns = nn.BatchNorm2d(out_features)

        self.skips = conv_block(in_features=in_features1,
                                out_features=out_features,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type=None,
                                activation=False)
        self.skips_2 = conv_block(in_features=in_features2,
                                  out_features=out_features,
                                  kernel_size=(1, 1),
                                  padding=(0, 0),
                                  norm_type=None,
                                  activation=False)
        # skips_3 用于处理通道数为 in_features // 2 的情况
        self.skips_3 = nn.Conv2d(in_features3, out_features,
                                 kernel_size=3, stride=2, dilation=2, padding=2)
        self.relu = nn.ReLU()

        self.gelu = nn.GELU()

    def forward(self, x_):
        x_low = x_[0]
        x = x_[1]
        x_high = x_[2]
        if x_high is not None:
            x_high = self.skips_3(x_high)
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low is not None:
            x_low = self.skips_2(x_low)
            x_low = F.interpolate(x_low, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1)
        x_skip = self.skips(x)
        x = self.skips(x)
        x = torch.chunk(x, 4, dim=1)
        if x_high is None:
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low is None:
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_high[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_high[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_high[3]), dim=1))
        else:
            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        x += x_skip
        x = self.bns(x)
        x = self.relu(x)

        return x


class DASI1(nn.Module):
    def __init__(self, in_features1,in_features2,in_features3, out_features,ri=0) -> None:
        super().__init__()
        self.r = ri
        self.bag = Bag()
        self.tail_conv = nn.Sequential(
            conv_block(in_features=out_features,
                       out_features=out_features,
                       kernel_size=(1, 1),
                       padding=(0, 0),
                       norm_type=None,
                       activation=False)
        )
        self.conv = nn.Sequential(
            conv_block(in_features=out_features // 2,
                       out_features=out_features // 4,
                       kernel_size=(1, 1),
                       padding=(0, 0),
                       norm_type=None,
                       activation=False)
        )
        self.bns = nn.BatchNorm2d(out_features)

        self.skips = conv_block(in_features=in_features1,
                                out_features=out_features,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type=None,
                                activation=False)
        self.skips_2 = conv_block(in_features=in_features2,
                                  out_features=out_features,
                                  kernel_size=(1, 1),
                                  padding=(0, 0),
                                  norm_type=None,
                                  activation=False)
        # skips_3 用于处理通道数为 in_features // 2 的情况
        self.skips_3 = nn.Conv2d(in_features3, out_features,
                                 kernel_size=3, stride=2, dilation=2, padding=2)
        # self.skips_3 = nn.Conv2d(in_features//2, out_features,
        #                          kernel_size=3, stride=2, dilation=1, padding=1)
        self.relu = nn.ReLU()

        self.gelu = nn.GELU()

    def forward(self, x_):
        x_low = x_[0]
        x = x_[1]
        x_high = x_[2]
        if x_high is not None:
            x_high = self.skips_3(x_high)

        if x_low is not None:
            x_low = self.skips_2(x_low)
            x_low = F.interpolate(x_low, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)

        x_skip = self.skips(x)
        x = x_skip

        x0 = self.bag(x_low, x_high, x)

        x = self.tail_conv(x0)
        x += x_skip
        x = self.bns(x)
        x = self.relu(x)

        return x


# 通道‑空间注意模块
class ChannelSpatialAttention(nn.Module):
    """轻量级通道‑空间注意模块，借鉴 CBAM 设计。"""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        # 通道注意：采用全局平均池化和最大池化，两个 MLP 分支共享权重
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, channels)
        # 空间注意：使用 7×7 卷积聚合通道统计信息
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                      padding=spatial_kernel // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # ----- 通道注意 -----
        avg_pooled = self.avg_pool(x).view(b, c)
        max_pooled = self.max_pool(x).view(b, c)
        # 共享 MLP
        fc_shared = lambda t: self.fc2(F.relu(self.fc1(t)))
        channel_att = self.sigmoid(fc_shared(avg_pooled) + fc_shared(max_pooled)).view(b, c, 1, 1)
        x = x * channel_att
        # ----- 空间注意 -----
        avg_pool_spatial = torch.mean(x, dim=1, keepdim=True)
        max_pool_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial))
        x = x * spatial_att
        return x


class DASI1Plus(nn.Module):
    """
    改进版 DASI1 模块。
      1. 可学习权重的三尺度特征融合；
      2. 使用 3×3、5×5、7×7 深度可分卷积的多分支特征分发；
      3. 通道‑空间注意模块强化关键信息；
      4. 残差连接保留中层细节。

    参数：
      in_features1: 中层特征通道数（与 x_[1] 对应）
      in_features2: 低层特征通道数（与 x_[0] 对应）
      in_features3: 高层特征通道数（与 x_[2] 对应）
      out_features: 输出特征通道数
    """
    def __init__(self, in_features1, in_features2, in_features3, out_features, ri=0) -> None:
        super().__init__()
        # 统一各尺度特征的通道数
        self.conv_mid = conv_block(in_features=in_features1, out_features=out_features,
                                   kernel_size=(1, 1), padding=(0, 0), norm_type=None,
                                   activation=False)
        self.conv_low = conv_block(in_features=in_features2, out_features=out_features,
                                   kernel_size=(1, 1), padding=(0, 0), norm_type=None,
                                   activation=False)
        # 高层特征使用 stride=2 下采样
        self.conv_high = nn.Conv2d(in_features3, out_features, kernel_size=1, stride=2, bias=False)
        # 权重生成器：使用全局平均池化 + 两层 1×1 卷积生成三个权重
        reduction = max(1, out_features // 4)
        self.w_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_features, reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, 3, 1, bias=True)
        )
        # 多分支深度可分卷积（3×3、5×5、7×7）
        self.dw1 = nn.Conv2d(out_features, out_features, kernel_size=1, padding=0,
                             groups=out_features, bias=False)
        self.dw3 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1,
                             groups=out_features, bias=False)
        self.dw5 = nn.Conv2d(out_features, out_features, kernel_size=5, padding=2,
                             groups=out_features, bias=False)
        # 通道融合：将 identity 分支和三个深度卷积分支拼接后用 1×1 卷积降维
        self.channel_fuse = nn.Conv2d(out_features * 4, out_features, kernel_size=1, bias=False)
        # 通道‑空间注意
        self.cs_att = ChannelSpatialAttention(out_features, reduction=16, spatial_kernel=7)
        # 归一化 + 激活
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_):
        # x_[0]: 低层特征, x_[1]: 中层特征, x_[2]: 高层特征
        x_low, x_mid, x_high = x_
        # 处理高层
        if x_high is not None:
            f_high = self.conv_high(x_high)
            # 下采样后再上采样到中层尺寸
            f_high = F.interpolate(f_high, size=x_mid.shape[2:], mode='nearest')
        else:
            f_high = 0
        # 处理低层
        if x_low is not None:
            f_low = self.conv_low(x_low)
            f_low = F.interpolate(f_low, size=x_mid.shape[2:], mode='nearest')
        else:
            f_low = 0
        # 处理中层
        f_mid = self.conv_mid(x_mid)
        # 生成融合权重
        w = self.w_gen(f_mid)  # (B,3,1,1)
        w = F.relu(w)
        w_sum = w.sum(dim=1, keepdim=True) + 1e-6
        w = w / w_sum
        # 按权重加权求和融合三尺度特征
        fused = (w[:, 0:1] * f_low) + (w[:, 1:2] * f_mid) + (w[:, 2:3] * f_high)
        # 多分支深度卷积并连接
        branch_id = fused
        branch3  = self.dw1(fused)
        branch5  = self.dw3(fused)
        branch7  = self.dw5(fused)
        concat = torch.cat([branch_id, branch3, branch5, branch7], dim=1)
        out = self.channel_fuse(concat)
        # 残差连接：加回融合特征
        out = out + fused
        # 通道‑空间注意
        out = self.cs_att(out)
        # BN + ReLU
        out = self.bn(out)
        out = self.relu(out)
        # 添加来自中层的残差以保留细节
        out = out + f_mid
        return out


class ExpertFusion(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        # 初始化三个权重为 1，训练中自动调整
        self.weights = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, f_low: torch.Tensor, f_mid: torch.Tensor, f_high: torch.Tensor) -> torch.Tensor:
        # ReLU 约束权重非负，再归一化使其和为 1
        w = F.relu(self.weights)
        w = w / (w.sum() + 1e-6)
        return w[0] * f_low + w[1] * f_mid + w[2] * f_high


class DASI1MoE(nn.Module):
    """
    混合专家多尺度融合模块，通过 gating 网络动态选择多名专家的融合结果。

    参数：
      in_features1: 中层特征通道数（与 x_[1] 对应）
      in_features2: 低层特征通道数（与 x_[0] 对应）
      in_features3: 高层特征通道数（与 x_[2] 对应）
      out_features: 输出通道数
      num_experts: 专家数量，默认为 3
    """

    def __init__(self, in_features1, in_features2, in_features3,
                 out_features, num_experts: int = 3, ri: int = 0) -> None:
        super().__init__()
        # 统一各尺度特征至 out_features 通道
        self.conv_low = Conv(in_features1, out_features, k=1, s=2)
        self.conv_mid = Conv(in_features2, out_features, k=1)
        self.conv_high = Conv(in_features3, out_features, k=1)
        # 高层特征采用 1×1 卷积 + stride=2 下采样

        # 初始化若干个专家，每个专家各自学习 f_low、f_mid、f_high 的加权和
        self.experts = nn.ModuleList([ExpertFusion(out_features) for _ in range(num_experts)])
        # gating 网络：输入中层特征的全局平均池化信息，输出各专家的权重
        reduction = max(1, out_features // 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_features, reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, num_experts, 1, bias=True)
        )

        # 三分支卷积：(1×1, 3×3, 5×5)
        self.b1 = nn.Conv2d(out_features, out_features, kernel_size=1, padding=0, bias=False)
        self.dw3 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1,
                             groups=out_features, bias=False)
        self.dw5 = nn.Conv2d(out_features, out_features, kernel_size=5, padding=2,
                             groups=out_features, bias=False)

        # 通道融合：C*3 -> C
        self.channel_fuse = nn.Conv2d(out_features * 3, out_features, kernel_size=1, bias=False)

        # 通道‑空间注意模块 (之前已在 DASI1Plus 中实现，可直接复用)
        self.cs_att = ChannelSpatialAttention(out_features, reduction=16, spatial_kernel=7)
        # 批归一化和激活
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_):
        # 分别获取低层、中层、高层输入
        x_low, x_mid, x_high = x_
        # 先统一各特征的通道数和尺寸
        f_mid = self.conv_mid(x_mid)
        if x_low is not None:
            f_low = self.conv_low(x_low)
        else:
            f_low = torch.zeros_like(f_mid)
        if x_high is not None:
            f_high = self.conv_high(x_high)
            f_high = F.interpolate(f_high, size=x_mid.shape[2:], mode='nearest')
        else:
            f_high = torch.zeros_like(f_mid)
        # gating 网络为每个专家生成权重 (B,num_experts,1,1)
        gate_logits = self.gate(f_mid)
        gate_logits = gate_logits.view(gate_logits.shape[0], gate_logits.shape[1])
        gate_weights = F.softmax(gate_logits, dim=1)  # (B, num_experts)
        # 每个专家融合 f_low、f_mid、f_high
        fused_outputs = [expert(f_low, f_mid, f_high) for expert in self.experts]
        # 堆叠为 (B,num_experts,C,H,W)，再按 gating 权重加权求和
        stacked = torch.stack(fused_outputs, dim=0).permute(1, 0, 2, 3, 4)
        gate_weights_exp = gate_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fused = torch.sum(gate_weights_exp * stacked, dim=1)

        b1 = self.b1(fused)
        b3 = self.dw3(fused)
        b5 = self.dw5(fused)
        concat = torch.cat([b1, b3, b5], dim=1)
        out = self.channel_fuse(concat)
        out = out + fused

        # 通道‑空间注意
        out = self.cs_att(out)
        # BN + ReLU
        out = self.bn(out)
        out = self.relu(out)
        # 残差连接中层特征，保留细节
        out = out + f_mid
        return out


class SimplifiedMoE(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=3):
        super().__init__()
        # 核心1：多个专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU()
            ) for _ in range(num_experts)
        ])

        # 核心2：轻量级门控网络
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_experts, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 门控权重
        gate_weights = self.gate(x)  # (B, num_experts, 1, 1)

        # 专家输出加权
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(gate_weights.unsqueeze(2) * expert_outputs, dim=1)

        return output


class MoE(nn.Module):
    """
    输入顺序固定为 [P2, P3, P4]
    anchor_idx:
        0 -> 输出 P2
        1 -> 输出 P3
        2 -> 输出 P4
    """

    def __init__(self, c2, c3, c4, out_features, target_level=2, num_experts=3):
        super().__init__()
        self.target_level = target_level

        # 不再写死谁上下采样，先统一通道
        self.conv_p2 = Conv(c2, out_features, k=3, s=1)
        self.conv_p3 = Conv(c3, out_features, k=3, s=1)
        self.conv_p4 = Conv(c4, out_features, k=3, s=1)

        self.experts = nn.ModuleList([ExpertFusion(out_features) for _ in range(num_experts)])

        reduction = max(1, out_features // 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_features, reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, num_experts, 1, bias=True)
        )

        self.b1 = nn.Conv2d(out_features, out_features, kernel_size=1, padding=0, bias=False)
        self.dw3 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1,
                             groups=out_features, bias=False)
        self.dw5 = nn.Conv2d(out_features, out_features, kernel_size=5, padding=2,
                             groups=out_features, bias=False)

        self.channel_fuse = nn.Conv2d(out_features * 3, out_features, kernel_size=1, bias=False)
        self.cs_att = ChannelSpatialAttention(out_features, reduction=16, spatial_kernel=7)

        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def _resize_to(self, x, size):
        if x.shape[2:] == size:
            return x
        # 大图缩小
        elif x.shape[2] > size[0] or x.shape[3] > size[1]:
            return F.adaptive_avg_pool2d(x, size)
        # 小图放大
        else:
            return F.interpolate(x, size=size, mode='nearest')

    def forward(self, x_):
        p2, p3, p4 = x_

        f2 = self.conv_p2(p2)
        f3 = self.conv_p3(p3)
        f4 = self.conv_p4(p4)

        if self.target_level == 2:
            target_size = f2.shape[2:]
        elif self.target_level == 3:
            target_size = f3.shape[2:]
        else:  # self.target_level == 4
            target_size = f4.shape[2:]

        # 全部对齐到目标层尺寸
        f2 = self._resize_to(f2, target_size)
        f3 = self._resize_to(f3, target_size)
        f4 = self._resize_to(f4, target_size)

        if self.target_level == 2:
            anchor_feat = f2
        elif self.target_level == 3:
            anchor_feat = f3
        else:
            anchor_feat = f4

        # gate 用目标层特征来生成专家权重
        gate_logits = self.gate(anchor_feat)
        gate_logits = gate_logits.view(gate_logits.shape[0], gate_logits.shape[1])
        gate_weights = F.softmax(gate_logits, dim=1)

        fused_outputs = [expert(f2, f3, f4) for expert in self.experts]
        stacked = torch.stack(fused_outputs, dim=0).permute(1, 0, 2, 3, 4)
        gate_weights_exp = gate_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fused = torch.sum(gate_weights_exp * stacked, dim=1)

        b1 = self.b1(fused)
        b3 = self.dw3(fused)
        b5 = self.dw5(fused)
        concat = torch.cat([b1, b3, b5], dim=1)
        out = self.channel_fuse(concat)
        out = out + fused

        out = self.cs_att(out)
        out = self.bn(out)
        out = self.relu(out)

        # 残差回到目标层
        out = out + anchor_feat
        return out

if __name__ == "__main__":
    B, C, H, W = 1, 128, 40, 40  # batch size, channels, height, width
    in_features = 128
    out_features = 128

    model = MoE(64,128,256, 128, 4)

    x = torch.randn(B, in_features, H, W)
    x_low = torch.randn(B, in_features // 2, H * 2, W * 2)
    x_high = torch.randn(B, in_features * 2, H // 2, W // 2)

    out = model([x_low, x, x_high])
    print("Output shape:", out.shape)