"""Direct tri-scale mixture-of-experts feature fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ("DMoE",)


class DMoE(nn.Module):
    """Directly fuse P2, P3, and P4 at a selected target resolution."""

    def __init__(self, c2, c3, c4, out_c, target_level=3, reduction=8):
        super().__init__()
        if target_level not in (2, 3, 4):
            raise ValueError(f"target_level must be 2, 3, or 4, got {target_level}.")
        self.target_level = target_level

        self.proj_p2 = Conv(c2, out_c, k=1, s=1)
        self.proj_p3 = Conv(c3, out_c, k=1, s=1)
        self.proj_p4 = Conv(c4, out_c, k=1, s=1)

        self.scale_gate = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, bias=True),
        )

        self.expert_detail = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )
        self.expert_context = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=2, dilation=2, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

        hidden = max(out_c // reduction, 4)
        self.expert_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_c, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, bias=True),
        )
        self.out_conv = Conv(out_c, out_c, k=1, s=1)

    @staticmethod
    def _resize_feature(x, target_hw):
        """Resize with adaptive average pooling or nearest interpolation."""
        h, w = x.shape[2:]
        th, tw = target_hw
        if (h, w) == (th, tw):
            return x
        if h > th or w > tw:
            return F.adaptive_avg_pool2d(x, target_hw)
        return F.interpolate(x, size=target_hw, mode="nearest")

    def forward(self, x):
        p2, p3, p4 = x
        p2 = self.proj_p2(p2)
        p3 = self.proj_p3(p3)
        p4 = self.proj_p4(p4)

        reference = (p2, p3, p4)[self.target_level - 2]
        target_hw = reference.shape[2:]
        p2_t = self._resize_feature(p2, target_hw)
        p3_t = self._resize_feature(p3, target_hw)
        p4_t = self._resize_feature(p4, target_hw)

        base = p2_t + p3_t + p4_t
        statistics = torch.cat(
            (base.mean(dim=1, keepdim=True), base.amax(dim=1, keepdim=True)),
            dim=1,
        )
        scale_weights = self.scale_gate(statistics).softmax(dim=1)
        fused = (
            scale_weights[:, 0:1] * p2_t
            + scale_weights[:, 1:2] * p3_t
            + scale_weights[:, 2:3] * p4_t
        )

        detail = self.expert_detail(fused)
        context = self.expert_context(fused)
        expert_weights = self.expert_gate(fused).softmax(dim=1)
        out = expert_weights[:, 0:1] * detail + expert_weights[:, 1:2] * context
        return self.out_conv(out + fused) + reference
