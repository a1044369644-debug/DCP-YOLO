# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Paper-aligned implementations of AFRD, LCC-C3k2, and APFF.

The public class names use the formal terminology from the paper:

* :class:`AFRD` - Adaptive Feature Retention Downsampling
* :class:`LCC_C3k2` - Local Coordinate Calibration C3k2
* :class:`APFF` - Adaptive Pyramid Feature Fusion

The implementations intentionally follow the equations described in the paper
instead of preserving the behavior of the former DCED, DGCA_C3k2, and DMoE
classes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("AFRD", "LCCBlock", "LCC_C3k2", "APFF")


def _autopad(kernel_size: int | tuple[int, int], dilation: int = 1) -> int | tuple[int, int]:
    """Return padding that preserves the spatial size for an odd kernel."""
    if isinstance(kernel_size, tuple):
        return tuple(dilation * (size - 1) // 2 for size in kernel_size)
    return dilation * (kernel_size - 1) // 2


class CBS(nn.Module):
    """Convolution followed by batch normalization and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 1,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        activation: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            _autopad(kernel_size, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation is True:
            self.act = nn.SiLU(inplace=True)
        elif isinstance(activation, nn.Module):
            self.act = activation
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableCBS(nn.Module):
    """Depthwise 3x3 CBS followed by pointwise 1x1 CBS."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.depthwise = CBS(in_channels, in_channels, kernel_size, groups=in_channels)
        self.pointwise = CBS(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class EfficientChannelAttention(nn.Module):
    """ECA: GAP -> channel-neighborhood Conv1D -> Sigmoid."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"ECA kernel size must be a positive odd integer, got {kernel_size}.")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.channel_conv(weights).transpose(-1, -2).unsqueeze(-1)
        return x * weights.sigmoid()


class SpatialBranchGate(nn.Module):
    """Generate per-pixel Softmax weights from concatenated branch features."""

    def __init__(self, branch_count: int = 3, kernel_size: int = 7) -> None:
        super().__init__()
        self.branch_count = branch_count
        self.gate = nn.Conv2d(2, branch_count, kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.branch_count:
            raise ValueError(f"Expected {self.branch_count} branches, got {len(features)}.")
        stacked = torch.cat(tuple(features), dim=1)
        statistics = torch.cat(
            (stacked.mean(dim=1, keepdim=True), stacked.amax(dim=1, keepdim=True)),
            dim=1,
        )
        return self.gate(statistics).softmax(dim=1)


class AFRD(nn.Module):
    """Adaptive Feature Retention Downsampling.

    This version implements the reported paper equations, including
    ``DWConv(X_r) - AvgPool(X_r)`` for the high-frequency branch, ECA channel
    recalibration, and learnable smoothing before Space-to-Depth downsampling.

    The constructor retains a Conv-compatible parameter order so the module can
    replace the former DCED entry in an Ultralytics YAML model definition.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 2,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
        e: float = 0.5,
        alpha_init: float = 0.5,
    ) -> None:
        super().__init__()
        if s not in (1, 2):
            raise ValueError(f"AFRD only supports stride 1 or 2, got {s}.")
        if not 0.0 < alpha_init < 1.0:
            raise ValueError(f"alpha_init must be in (0, 1), got {alpha_init}.")
        if g != 1:
            raise ValueError("AFRD enhancement branches are depthwise; the compatibility argument g must be 1.")
        if p is not None and p != _autopad(k, d):
            raise ValueError("AFRD requires same-padding in its local-detail branch.")

        hidden_channels = max(int(c2 * e), 1)
        self.stride = s
        self.channel_projection = CBS(c1, hidden_channels, 1, activation=act)
        self.local_detail_branch = CBS(
            hidden_channels,
            hidden_channels,
            k,
            groups=hidden_channels,
            dilation=d,
            activation=act,
        )
        self.context_branch = CBS(
            hidden_channels,
            hidden_channels,
            3,
            groups=hidden_channels,
            dilation=2,
            activation=act,
        )
        self.high_frequency_branch = CBS(
            hidden_channels,
            hidden_channels,
            3,
            groups=hidden_channels,
            activation=act,
        )
        self.branch_gate = SpatialBranchGate(branch_count=3, kernel_size=7)
        self.channel_attention = EfficientChannelAttention(kernel_size=3)

        alpha_logit = math.log(alpha_init / (1.0 - alpha_init))
        self.smoothing_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))
        projection_in_channels = hidden_channels * 4 if s == 2 else hidden_channels
        self.output_projection = CBS(projection_in_channels, c2, 1, activation=act)

    @property
    def smoothing_coefficient(self) -> torch.Tensor:
        """Return the learnable smoothing coefficient alpha in the range (0, 1)."""
        return self.smoothing_logit.sigmoid()

    @staticmethod
    def _pad_to_even(x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_height, pad_width = height % 2, width % 2
        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")
        return x

    @staticmethod
    def _space_to_depth(x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                x[..., 0::2, 0::2],
                x[..., 1::2, 0::2],
                x[..., 0::2, 1::2],
                x[..., 1::2, 1::2],
            ),
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.channel_projection(x)
        local_feature = self.local_detail_branch(reduced)
        context_feature = self.context_branch(reduced)

        low_frequency = F.avg_pool2d(reduced, kernel_size=3, stride=1, padding=1)
        high_frequency = self.high_frequency_branch(reduced) - low_frequency

        branch_weights = self.branch_gate((local_feature, context_feature, high_frequency))
        fused = (
            branch_weights[:, 0:1] * local_feature
            + branch_weights[:, 1:2] * context_feature
            + branch_weights[:, 2:3] * high_frequency
        )
        recalibrated = self.channel_attention(fused)

        if self.stride == 2:
            alpha = self.smoothing_coefficient.to(dtype=recalibrated.dtype)
            smoothed = F.avg_pool2d(recalibrated, kernel_size=3, stride=1, padding=1)
            recalibrated = (1.0 - alpha) * recalibrated + alpha * smoothed
            recalibrated = self._space_to_depth(self._pad_to_even(recalibrated))

        return self.output_projection(recalibrated)


class DualPoolingChannelAttention(nn.Module):
    """DPCA: [GAP(X), GMP(X)] -> CBS 1x1 -> Sigmoid."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mapping = CBS(2 * channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptors = torch.cat(
            (F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)),
            dim=1,
        )
        return x * self.mapping(descriptors).sigmoid()


class CrossAxisSpatialAttention(nn.Module):
    """CASA with 3x1/1x3 directional convolution and CBS fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.height_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=channels,
            bias=False,
        )
        self.width_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=channels,
            bias=False,
        )
        self.fusion = CBS(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height_descriptor = x.mean(dim=3, keepdim=True)
        width_descriptor = x.mean(dim=2, keepdim=True)
        height_response = self.height_conv(height_descriptor)
        width_response = self.width_conv(width_descriptor)
        spatial_mask = self.fusion(height_response + width_response).sigmoid()
        return x * spatial_mask


class LCCBlock(nn.Module):
    """Local Coordinate Calibration block matching the paper topology."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError(f"LCCBlock requires an even channel count for an equal split, got {channels}.")

        self.transform_channels = channels // 2
        self.local_convolution = CBS(self.transform_channels, self.transform_channels, 3)
        self.local_dsconv = DepthwiseSeparableCBS(self.transform_channels, self.transform_channels, 3)
        self.channel_attention = DualPoolingChannelAttention(self.transform_channels)
        self.spatial_attention = CrossAxisSpatialAttention(self.transform_channels)
        self.transform_fusion = DepthwiseSeparableCBS(
            2 * self.transform_channels,
            self.transform_channels,
            3,
        )
        self.output_fusion = CBS(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transform_input, bypass = x.split(self.transform_channels, dim=1)
        local_feature = self.local_dsconv(self.local_convolution(transform_input))
        channel_calibrated = self.channel_attention(local_feature)
        coordinate_calibrated = self.spatial_attention(channel_calibrated)
        transformed = self.transform_fusion(torch.cat((coordinate_calibrated, local_feature), dim=1))
        return self.output_fusion(torch.cat((transformed, bypass), dim=1))


class LCC_C3k2(nn.Module):
    """C3k2 whose internal feature-extraction units are LCC blocks.

    The argument order places ``e`` immediately after ``shortcut`` so a YAML
    entry such as ``[256, False, 0.25]`` correctly uses 0.25 as the expansion
    ratio. The paper-aligned LCC block has no internal residual connection;
    therefore ``shortcut=True`` is rejected instead of silently changing the
    published computation.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        if shortcut:
            raise ValueError("The paper-aligned LCC block does not define an internal shortcut.")
        if n < 1:
            raise ValueError(f"LCC_C3k2 requires at least one LCC block, got n={n}.")

        self.hidden_channels = max(int(c2 * e), 1)
        if self.hidden_channels % 2:
            raise ValueError(
                "The expansion ratio must produce an even hidden channel count; "
                f"got int({c2} * {e}) = {self.hidden_channels}."
            )
        self.input_projection = CBS(c1, 2 * self.hidden_channels, 1)
        self.blocks = nn.ModuleList(LCCBlock(self.hidden_channels) for _ in range(n))
        self.output_projection = CBS((2 + n) * self.hidden_channels, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = list(self.input_projection(x).chunk(2, dim=1))
        for block in self.blocks:
            features.append(block(features[-1]))
        return self.output_projection(torch.cat(features, dim=1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        features = list(
            self.input_projection(x).split((self.hidden_channels, self.hidden_channels), dim=1)
        )
        for block in self.blocks:
            features.append(block(features[-1]))
        return self.output_projection(torch.cat(features, dim=1))


class APFF(nn.Module):
    """Adaptive Pyramid Feature Fusion for direct P2/P3/P4 aggregation."""

    def __init__(
        self,
        c2: int,
        c3: int,
        c4: int,
        out_c: int,
        target_level: int = 3,
    ) -> None:
        super().__init__()
        if target_level not in (2, 3, 4):
            raise ValueError(f"target_level must be 2, 3, or 4, got {target_level}.")

        self.target_level = target_level
        self.projections = nn.ModuleList(
            (CBS(c2, out_c, 1), CBS(c3, out_c, 1), CBS(c4, out_c, 1))
        )
        self.scale_gate = nn.Conv2d(2, 3, kernel_size=7, padding=3, bias=True)
        self.detail_branch = nn.Sequential(
            CBS(out_c, out_c, 3, groups=out_c),
            CBS(out_c, out_c, 1),
        )
        self.context_branch = nn.Sequential(
            CBS(out_c, out_c, 3, groups=out_c, dilation=2),
            CBS(out_c, out_c, 1),
        )
        self.expert_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            CBS(out_c, 2, 1),
        )
        self.output_mapping = CBS(out_c, out_c, 3)

    @staticmethod
    def _resize_feature(x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        height, width = x.shape[-2:]
        target_height, target_width = target_size
        if (height, width) == target_size:
            return x
        if height > target_height or width > target_width:
            return F.adaptive_avg_pool2d(x, target_size)
        return F.interpolate(x, size=target_size, mode="nearest")

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(x) != 3:
            raise ValueError(f"APFF expects the three features P2, P3, and P4, got {len(x)} inputs.")

        projected = [projection(feature) for projection, feature in zip(self.projections, x)]
        target_index = self.target_level - 2
        target_size = projected[target_index].shape[-2:]
        aligned = [self._resize_feature(feature, target_size) for feature in projected]
        target_residual = aligned[target_index]

        scale_stack = torch.cat(aligned, dim=1)
        scale_statistics = torch.cat(
            (scale_stack.mean(dim=1, keepdim=True), scale_stack.amax(dim=1, keepdim=True)),
            dim=1,
        )
        scale_weights = self.scale_gate(scale_statistics).softmax(dim=1)
        fused = sum(
            scale_weights[:, index : index + 1] * feature
            for index, feature in enumerate(aligned)
        )

        detail_feature = self.detail_branch(fused)
        context_feature = self.context_branch(fused)
        expert_weights = self.expert_gate(fused).softmax(dim=1)
        combined = (
            fused
            + expert_weights[:, 0:1] * detail_feature
            + expert_weights[:, 1:2] * context_feature
            + target_residual
        )
        return self.output_mapping(combined)
