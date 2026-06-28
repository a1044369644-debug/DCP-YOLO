import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

def autopad(k, p=None, d=1):
    if p is not None:
        return p
    if isinstance(k, tuple):
        if isinstance(d, tuple):
            return tuple(((kk - 1) * dd) // 2 for kk, dd in zip(k, d))
        return tuple(((kk - 1) * d) // 2 for kk in k)
    return ((k - 1) * d) // 2


def resize_feature(x, target_size):
    if x.shape[2:] == target_size:
        return x
    if x.shape[2] > target_size[0] or x.shape[3] > target_size[1]:
        return F.adaptive_avg_pool2d(x, target_size)
    return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = nn.SiLU(inplace=True)
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSAttnLite(nn.Module):
    """
    轻量级通道-空间注意
    如果你已经有自己的 ChannelSpatialAttention，也可以直接替换这里
    """

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        hidden = max(8, channels // reduction)

        self.ca_avg_1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.ca_avg_2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.ca_max_1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.ca_max_2 = nn.Conv2d(hidden, channels, 1, bias=True)

        self.sa = nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)

        ca = self.ca_avg_2(self.relu(self.ca_avg_1(avg))) + self.ca_max_2(self.relu(self.ca_max_1(mx)))
        x = x * self.sigmoid(ca)

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid(self.sa(torch.cat([avg_map, max_map], dim=1)))
        return x * sa


class MidAlign(nn.Module):
    """
    mid: H x W -> H x W
    只做通道对齐 + 轻量校准
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = ConvBNAct(c1, c2, k=1, act=False)
        self.refine = ConvBNAct(c2, c2, k=3, g=c2, act=True)
        self.gamma = nn.Parameter(torch.ones(1, c2, 1, 1))

    def forward(self, x, target_size=None):
        x = self.proj(x)
        if target_size is not None and x.shape[2:] != target_size:
            x = resize_feature(x, target_size)
        x = self.refine(x)
        return x * self.gamma


class LowToMidAlign(nn.Module):
    """
    low: 2H x 2W -> H x W
    显式下采样到 mid 尺度
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = ConvBNAct(c1, c2, k=1, act=False)
        # 先学习式下采样
        self.down = ConvBNAct(c2, c2, k=3, s=2, g=c2, act=True)
        self.refine = ConvBNAct(c2, c2, k=3, g=c2, act=True)
        self.gamma = nn.Parameter(torch.ones(1, c2, 1, 1))

    def forward(self, x, target_size):
        x = self.proj(x)
        if x.shape[2:] == target_size or x.shape[2] < target_size[0] or x.shape[3] < target_size[1]:
            x = resize_feature(x, target_size)
            x = self.refine(x)
            return x * self.gamma
        x = self.down(x)  # 2H x 2W -> H x W（理想情况）
        if x.shape[2:] != target_size:
            x = resize_feature(x, target_size)
        x = self.refine(x)
        return x * self.gamma


class HighToMidAlign(nn.Module):
    """
    high: H/2 x W/2 -> H x W
    显式上采样到 mid 尺度
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = ConvBNAct(c1, c2, k=1, act=False)
        self.refine_before = ConvBNAct(c2, c2, k=3, g=c2, act=True)
        self.refine_after = ConvBNAct(c2, c2, k=3, g=c2, act=True)
        self.gamma = nn.Parameter(torch.ones(1, c2, 1, 1))

    def forward(self, x, target_size):
        x = self.proj(x)
        x = self.refine_before(x)
        x = resize_feature(x, target_size)
        x = self.refine_after(x)
        return x * self.gamma


class DetailExpert(nn.Module):
    """偏局部纹理/边缘"""

    def __init__(self, c):
        super().__init__()
        self.dw = ConvBNAct(c, c, k=3, g=c, act=True)
        self.pw = ConvBNAct(c, c, k=1, act=False)

    def forward(self, x):
        high_pass = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        y = self.dw(high_pass) + x
        return self.pw(y)


class ContextExpert(nn.Module):
    """偏大感受野上下文，带一点 LSK 风格"""

    def __init__(self, c):
        super().__init__()
        self.dw5 = ConvBNAct(c, c, k=5, g=c, act=True)
        self.dw7d3 = ConvBNAct(c, c, k=7, d=3, g=c, act=True)  # 更大有效感受野
        self.mix = ConvBNAct(2 * c, c, k=1, act=False)

        hidden = max(8, c // 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.dw5(x)
        x2 = self.dw7d3(x)
        y = self.mix(torch.cat([x1, x2], dim=1))
        return y * self.gate(x) + x


class FrequencyExpert(nn.Module):
    """偏高低频分解，不依赖额外小波库"""

    def __init__(self, c):
        super().__init__()
        self.hp_dw = ConvBNAct(c, c, k=3, g=c, act=True)
        self.lp_pw = ConvBNAct(c, c, k=1, act=False)
        self.out_pw = ConvBNAct(c, c, k=1, act=False)

    def forward(self, x):
        low_freq = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        high_freq = x - low_freq
        y = self.hp_dw(high_freq) + self.lp_pw(low_freq)
        return self.out_pw(y)


class SAHSEF(nn.Module):
    """
    Scale-Aligned Heterogeneous Sparse Expert Fusion

    相比旧版 DASI1MoE：
    1) 先做跨尺度对齐
    2) 再做尺度加权
    3) 再做异构专家 + top-k 稀疏路由
    4) 最后做轻量重组与注意增强
    """

    def __init__(self, in_features1, in_features2, in_features3,
                 out_features, target_level: int = 3, num_experts: int = 3, topk: int = 2, ri: int = 0):
        super().__init__()
        assert target_level in [2, 3, 4]
        assert num_experts == 3, "当前实现固定为 3 个异构专家：detail / context / frequency"

        self.out_features = out_features
        self.target_level = target_level
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        # 三路特征对齐到同一通道数和空间尺度
        self.align_low = LowToMidAlign(in_features1, out_features)
        self.align_mid = MidAlign(in_features2, out_features)
        self.align_high = HighToMidAlign(in_features3, out_features)

        router_hidden = max(16, out_features // 2)

        # 路由描述子 = GAP(low) + GAP(mid) + GAP(high) + GMP(mid)
        self.scale_router = nn.Sequential(
            nn.Linear(out_features * 4, router_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(router_hidden, 3)
        )
        self.expert_router = nn.Sequential(
            nn.Linear(out_features * 4, router_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(router_hidden, num_experts)
        )

        # 3个异构专家
        self.experts = nn.ModuleList([
            DetailExpert(out_features),
            ContextExpert(out_features),
            FrequencyExpert(out_features),
        ])

        # 后端重组：1x1 / 3x3 DW / strip conv
        self.branch1 = ConvBNAct(out_features, out_features, k=1, act=False)
        self.branch3 = ConvBNAct(out_features, out_features, k=3, g=out_features, act=False)
        self.branch_strip = nn.Sequential(
            ConvBNAct(out_features, out_features, k=(1, 5), g=out_features, act=True),
            ConvBNAct(out_features, out_features, k=(5, 1), g=out_features, act=False),
        )
        self.channel_fuse = ConvBNAct(out_features * 3, out_features, k=1, act=False)

        self.cs_att = CSAttnLite(out_features, reduction=16, spatial_kernel=7)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = nn.ReLU(inplace=True)

        # 可选辅助损失
        self.aux_loss = None
        self.last_scale_weights = None
        self.last_expert_weights = None

    @staticmethod
    def _pool_desc(x):
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    @staticmethod
    def _max_desc(x):
        return F.adaptive_max_pool2d(x, 1).flatten(1)

    def _topk_softmax(self, logits):
        if not logits.is_floating_point():
            logits = logits.float()

        if self.topk >= self.num_experts:
            return F.softmax(logits, dim=1, dtype=torch.float32).to(logits.dtype)

        topk_val, topk_idx = torch.topk(logits, k=self.topk, dim=1)
        masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
        masked.scatter_(1, topk_idx, topk_val)

        out = F.softmax(masked, dim=1, dtype=torch.float32)
        return out.to(logits.dtype)
    def _calc_balance_loss(self, gate_weights):
        # 轻量负载均衡，防止专家长期塌缩到单一路
        importance = gate_weights.mean(dim=0)
        target = torch.full_like(importance, 1.0 / self.num_experts)
        return F.mse_loss(importance, target)

    def forward(self, x_):
        x_low, x_mid, x_high = x_
        target_size = x_mid.shape[2:]  # mid 的 H x W

        target = {2: x_low, 3: x_mid, 4: x_high}[self.target_level]
        if target is not None:
            target_size = target.shape[2:]
        f_mid = self.align_mid(x_mid, target_size)

        if x_low is not None:
            f_low = self.align_low(x_low, target_size)  # 2H x 2W -> H x W
        else:
            f_low = torch.zeros_like(f_mid)

        if x_high is not None:
            f_high = self.align_high(x_high, target_size)  # H/2 x W/2 -> H x W
        else:
            f_high = torch.zeros_like(f_mid)

        target_feat = {2: f_low, 3: f_mid, 4: f_high}[self.target_level]

        # 先做三尺度基础融合比例预测
        route_desc = torch.cat([
            self._pool_desc(f_low),
            self._pool_desc(f_mid),
            self._pool_desc(f_high),
            self._max_desc(f_mid)
        ], dim=1)

        scale_weights = F.softmax(self.scale_router(route_desc), dim=1)  # (B, 3)

        fused_base = (
                scale_weights[:, 0].view(-1, 1, 1, 1) * f_low +
                scale_weights[:, 1].view(-1, 1, 1, 1) * f_mid +
                scale_weights[:, 2].view(-1, 1, 1, 1) * f_high
        )

        # 稀疏专家路由，只激活 top-k
        expert_logits = self.expert_router(route_desc)
        expert_weights = self._topk_softmax(expert_logits)  # (B, E)

        expert_outs = [expert(fused_base) for expert in self.experts]
        expert_outs = torch.stack(expert_outs, dim=1)  # (B, E, C, H, W)

        fused = torch.sum(
            expert_outs * expert_weights[:, :, None, None, None],
            dim=1
        )

        # 后端轻量重组
        b1 = self.branch1(fused)
        b3 = self.branch3(fused)
        bs = self.branch_strip(fused)

        out = self.channel_fuse(torch.cat([b1, b3, bs], dim=1))
        out = out + fused

        out = self.cs_att(out)
        out = self.bn(out)
        out = self.act(out)

        # 保留中层细节
        out = out + target_feat

        # 训练时可选使用

        return out



class DMoE(nn.Module):
    """
    Direct Tri-scale MoE Fusion
    不使用外部上采样+拼接
    直接输入最新的 P2/P3/P4，输出指定目标尺度(P2/P3/P4)
    """

    def __init__(self, c2, c3, c4, out_c, target_level=3, reduction=8):
        super().__init__()
        assert target_level in [2, 3, 4]
        self.target_level = target_level

        # 1. 通道统一
        self.proj_p2 = Conv(c2, out_c, k=1, s=1)
        self.proj_p3 = Conv(c3, out_c, k=1, s=1)
        self.proj_p4 = Conv(c4, out_c, k=1, s=1)

        # 2. 三尺度内容权重
        self.scale_gate = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, bias=True)
        )

        # 3. 双专家
        self.expert_detail = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

        self.expert_context = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=2, dilation=2, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

        hidden = max(out_c // reduction, 4)
        self.expert_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_c, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, bias=True)
        )

        self.out_conv = Conv(out_c, out_c, k=1, s=1)

    def _resize_feature(self, x, target_hw):
        """
        尺度对齐：
        - 缩小：自适应平均池化
        - 放大：最近邻插值
        """
        h, w = x.shape[2:]
        th, tw = target_hw

        if (h, w) == (th, tw):
            return x
        elif h > th or w > tw:
            return F.adaptive_avg_pool2d(x, target_hw)
        else:
            return F.interpolate(x, size=target_hw, mode='nearest')

    def forward(self, x):
        p2, p3, p4 = x

        p2 = self.proj_p2(p2)
        p3 = self.proj_p3(p3)
        p4 = self.proj_p4(p4)

        # 目标尺度
        if self.target_level == 2:
            ref = p2
            target_hw = p2.shape[2:]
        elif self.target_level == 3:
            ref = p3
            target_hw = p3.shape[2:]
        else:
            ref = p4
            target_hw = p4.shape[2:]

        # 全部直接对齐到目标尺度
        p2_t = self._resize_feature(p2, target_hw)
        p3_t = self._resize_feature(p3, target_hw)
        p4_t = self._resize_feature(p4, target_hw)

        # -------- 三尺度动态加权 --------
        base = p2_t + p3_t + p4_t
        avg = torch.mean(base, dim=1, keepdim=True)
        mx, _ = torch.max(base, dim=1, keepdim=True)
        stat = torch.cat([avg, mx], dim=1)

        scale_w = self.scale_gate(stat)     # [B,3,H,W]
        scale_w = F.softmax(scale_w, dim=1)

        fused = (
            scale_w[:, 0:1] * p2_t +
            scale_w[:, 1:2] * p3_t +
            scale_w[:, 2:3] * p4_t
        )

        # -------- 轻量 MoE 双专家 --------
        e1 = self.expert_detail(fused)
        e2 = self.expert_context(fused)

        ew = self.expert_gate(fused)        # [B,2,1,1]
        ew = F.softmax(ew, dim=1)

        out = ew[:, 0:1] * e1 + ew[:, 1:2] * e2
        out = out + fused

        # 输出映射 + 目标层残差
        out = self.out_conv(out)
        out = out + ref

        return out
