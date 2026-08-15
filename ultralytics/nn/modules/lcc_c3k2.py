# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Local Coordinate Calibration C3k2 module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("LCC_C3k2",)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 CBS followed by pointwise 1x1 CBS."""

    def __init__(self, c1: int, c2: int, k: int = 3) -> None:
        super().__init__()
        self.depthwise = Conv(c1, c1, k, 1, g=c1)
        self.pointwise = Conv(c1, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class DualPoolingChannelAttention(nn.Module):
    """DPCA: [GAP(X), GMP(X)] -> CBS 1x1 -> Sigmoid."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mapping = Conv(2 * channels, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptors = torch.cat(
            (F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)),
            dim=1,
        )
        return x * self.mapping(descriptors).sigmoid()


class CrossAxisSpatialAttention(nn.Module):
    """CASA with 3x1 and 1x3 directional convolutions and CBS fusion."""

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
        self.fusion = Conv(channels, channels, 1, 1)

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
            raise ValueError(f"LCCBlock requires an even channel count, got {channels}.")

        self.transform_channels = channels // 2
        self.local_convolution = Conv(self.transform_channels, self.transform_channels, 3, 1)
        self.local_dsconv = DepthwiseSeparableConv(self.transform_channels, self.transform_channels, 3)
        self.channel_attention = DualPoolingChannelAttention(self.transform_channels)
        self.spatial_attention = CrossAxisSpatialAttention(self.transform_channels)
        self.transform_fusion = DepthwiseSeparableConv(
            2 * self.transform_channels,
            self.transform_channels,
            3,
        )
        self.output_fusion = Conv(channels, channels, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transform_input, bypass = x.split(self.transform_channels, dim=1)
        local_feature = self.local_dsconv(self.local_convolution(transform_input))
        channel_calibrated = self.channel_attention(local_feature)
        coordinate_calibrated = self.spatial_attention(channel_calibrated)
        transformed = self.transform_fusion(torch.cat((coordinate_calibrated, local_feature), dim=1))
        return self.output_fusion(torch.cat((transformed, bypass), dim=1))


class LCC_C3k2(nn.Module):
    """C3k2 whose internal feature-extraction units are LCC blocks.

    ``e`` immediately follows ``shortcut`` so a YAML entry such as
    ``[256, False, 0.25]`` correctly uses 0.25 as the expansion ratio.
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
        self.input_projection = Conv(c1, 2 * self.hidden_channels, 1, 1)
        self.blocks = nn.ModuleList(LCCBlock(self.hidden_channels) for _ in range(n))
        self.output_projection = Conv((2 + n) * self.hidden_channels, c2, 1, 1)

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
