# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math
from pytorch_wavelets import DWTForward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "CBAM",
    "CED",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
    "DWTConcatIgm",
    "DWTLSK",
    "CGMDown",
    "CGWTDown",
    "DCED",
    "SFDDown",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]: i[0] + 1, i[1]: i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SimpleBranch(nn.Module):
    """
    有序分支：简单一点
    先用 1 个普通 3x3 卷积
    """

    def __init__(self, c=3):
        super().__init__()
        self.block = ConvBNAct(c, c, k=3, s=1)

    def forward(self, x):
        return self.block(x)


class ComplexBranch(nn.Module):
    """
    无序分支：复杂一点
    先用 2 个普通 3x3 卷积
    """

    def __init__(self, c=3):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(c, c, k=3, s=1),
            ConvBNAct(c, c, k=3, s=1),
        )

    def forward(self, x):
        return self.block(x)


class DWTConcatIgm(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1=3, c2=32, k=3, s=2, p=None, g=1, d=1, act=True, wave='haar'):
        super(DWTConcatIgm, self).__init__()

        # 这版默认输入是RGB图像
        assert c1 == 3, "当前这版默认输入为RGB图像，所以 c1 必须为 3"

        self.c1 = c1
        self.c2 = c2

        # 单层DWT
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)

        # DWT拼接后通道数 = 4 * c1 = 12
        dwt_channels = 4 * c1

        # 用于生成 fore_weight
        # groups=4 表示每3通道一组分别卷积
        self.m_conv = nn.Conv2d(
            dwt_channels, dwt_channels,
            kernel_size=k, stride=1,
            padding=autopad(k, p, 1),
            groups=4, bias=True
        )

        self.register_buffer(
            'mask',
            self._create_diagonal_mask(k, dwt_channels, dwt_channels, g=4, mode='both')
        )

        self.sigmoid = nn.Sigmoid()

        # 4个有序分支（每组3通道）
        self.simple_branches = nn.ModuleList([SimpleBranch(c1) for _ in range(4)])

        # 4个无序分支（每组3通道）
        self.complex_branches = nn.ModuleList([ComplexBranch(c1) for _ in range(4)])

        # 8个可学习变量：4组 × 2个分支
        # alpha[i, 0] -> 第i组有序分支权重
        # alpha[i, 1] -> 第i组无序分支权重
        self.alpha = nn.Parameter(torch.ones(4, 2))

        # 拼接后再映射到输出通道 c2
        self.fuse = nn.Sequential(
            nn.Conv2d(dwt_channels, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # shortcut 分支：原图下采样后映射到 c2
        self.shortcut = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def _create_diagonal_mask(self, k, c1, c2, g=1, mode='both'):
        """
        对 grouped conv 的卷积核做掩码
        m_conv.weight 的形状为 [c2, c1//g, k, k]
        这里当 c1=c2=12, g=4 时 -> [12, 3, k, k]
        """
        mask = torch.ones((c2, c1 // g, k, k))
        idx = torch.arange(k)

        if mode in (1, 'main'):
            mask[:, :, idx, idx] = 0
        elif mode in (2, 'anti'):
            mask[:, :, idx, k - 1 - idx] = 0
        elif mode in (3, 'both', 'x'):
            mask[:, :, idx, idx] = 0
            mask[:, :, idx, k - 1 - idx] = 0
        else:
            raise ValueError(f"Unsupported mode={mode}")

        return mask

    def _masked_conv_forward(self, x):
        # 每次 forward 都乘 mask
        weight = self.m_conv.weight * self.mask
        return F.conv2d(
            x, weight, self.m_conv.bias,
            stride=self.m_conv.stride,
            padding=self.m_conv.padding,
            dilation=self.m_conv.dilation,
            groups=self.m_conv.groups
        )

    def forward(self, x):
        # x: [B, 3, H, W]

        # 1. DWT分解
        yl, yh = self.dwt(x)
        # yl: [B, 3, H/2, W/2]
        # yh[0]: [B, 3, 3, H/2, W/2]

        lh = yh[0][:, :, 0, :, :]  # [B, 3, H/2, W/2]
        hl = yh[0][:, :, 1, :, :]
        hh = yh[0][:, :, 2, :, :]

        # 2. 拼成4个分量，共12通道
        out = torch.cat([yl, lh, hl, hh], dim=1)  # [B, 12, H/2, W/2]

        # 3. 得到有序图像 fore_img 和无序图像 dif_img
        forecast = self._masked_conv_forward(out)  # [B, 12, H/2, W/2]
        fore_weight = self.sigmoid(forecast)

        fore_img = out * fore_weight
        dif_img = out - fore_img

        # 4. 每3个通道分为一个分量，共4组
        fore_parts = fore_img.chunk(4, dim=1)  # 4个 [B, 3, H/2, W/2]
        dif_parts = dif_img.chunk(4, dim=1)  # 4个 [B, 3, H/2, W/2]

        # 5. 有序分支走简单网络，无序分支走复杂网络
        alpha = torch.softmax(self.alpha, dim=1)  # [4, 2]

        fused_parts = []
        for i in range(4):
            fore_feat = self.simple_branches[i](fore_parts[i])  # [B, 3, H/2, W/2]
            dif_feat = self.complex_branches[i](dif_parts[i])  # [B, 3, H/2, W/2]

            fused_i = alpha[i, 0] * fore_feat + alpha[i, 1] * dif_feat
            fused_parts.append(fused_i)

        # 6. 拼接回去
        y = torch.cat(fused_parts, dim=1)  # [B, 12, H/2, W/2]

        # 7. 输出映射 + shortcut
        y = self.fuse(y)  # [B, c2, H/2, W/2]
        shortcut = self.shortcut(x)  # [B, c2, H/2, W/2]

        return y + shortcut


class DWTLSK(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1=3, c2=32, k=3, s=2, p=None, g=1, d=1, act=True, wave='haar'):
        super(DWTLSK, self).__init__()

        # 这版默认输入是RGB图像

        self.c1 = c1
        self.c2 = c2

        # 单层DWT
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)

        # DWT拼接后通道数 = 4 * c1 = 12
        dwt_channels = 4 * c1
        self.lsk_gate = LSKGate(dwt_channels)
        # 4个有序分支（每组3通道）
        self.simple_branches = nn.ModuleList([SimpleBranch(c1) for _ in range(4)])

        # 4个无序分支（每组3通道）
        self.complex_branches = nn.ModuleList([ComplexBranch(c1) for _ in range(4)])

        # 8个可学习变量：4组 × 2个分支
        # alpha[i, 0] -> 第i组有序分支权重
        # alpha[i, 1] -> 第i组无序分支权重
        self.alpha = nn.Parameter(torch.ones(4, 2))

        # 拼接后再映射到输出通道 c2
        self.fuse = nn.Sequential(
            nn.Conv2d(dwt_channels, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # shortcut 分支：原图下采样后映射到 c2
        self.shortcut = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        yl, yh = self.dwt(x)

        lh = yh[0][:, :, 0, :, :]
        hl = yh[0][:, :, 1, :, :]
        hh = yh[0][:, :, 2, :, :]

        out = torch.cat([yl, lh, hl, hh], dim=1)  # [B, 12, H/2, W/2]

        # 用 LSK 生成 fore_weight
        fore_weight = self.lsk_gate(out)

        fore_img = out * fore_weight
        dif_img = out - fore_img

        fore_parts = fore_img.chunk(4, dim=1)
        dif_parts = dif_img.chunk(4, dim=1)

        alpha = torch.softmax(self.alpha, dim=1)

        fused_parts = []
        for i in range(4):
            fore_feat = self.simple_branches[i](fore_parts[i])
            dif_feat = self.complex_branches[i](dif_parts[i])

            fused_i = alpha[i, 0] * fore_feat + alpha[i, 1] * dif_feat
            fused_parts.append(fused_i)

        y = torch.cat(fused_parts, dim=1)
        y = self.fuse(y)
        shortcut = self.shortcut(x)

        return y + shortcut


class LSKGate(nn.Module):
    def __init__(self, channels):
        super().__init__()

        mid_channels = max(channels // 2, 4)

        # 新增：3x3 掩码卷积（建议用 depthwise，开销小）
        self.mask3 = MaskConv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.dw3 = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        # 原来的两条大核分支
        self.dw5 = nn.Conv2d(
            channels, channels,
            kernel_size=5, padding=2,
            groups=channels, bias=False
        )
        self.dw7_d3 = nn.Conv2d(
            channels, channels,
            kernel_size=7, padding=9, dilation=3,
            groups=channels, bias=False
        )

        # 三个分支分别压缩通道
        self.pw0 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.pw1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)

        # 原来输出2个选择图，现在输出3个
        self.selector = nn.Conv2d(2, 3, kernel_size=7, padding=3, bias=True)

        # 投影回原通道
        self.proj = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        # 三层递进式特征
        a0 = self.dw3(x)  # 局部差异/边缘
        a1 = self.dw5(a0)  # 中等感受野
        a2 = self.dw7_d3(a1)  # 更大感受野

        # 各自压缩
        a0 = self.pw0(a0)
        a1 = self.pw1(a1)
        a2 = self.pw2(a2)

        # 聚合统计信息
        a = torch.cat([a0, a1, a2], dim=1)  # [B, 3*mid, H, W]
        avg = torch.mean(a, dim=1, keepdim=True)
        mx, _ = torch.max(a, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)  # [B, 2, H, W]

        # 生成3个分支权重
        s = self.selector(s)  # [B, 3, H, W]
        s = torch.softmax(s, dim=1)  # 更适合多分支竞争选择

        # 三分支自适应融合
        out = (
                a0 * s[:, 0:1, :, :] +
                a1 * s[:, 1:2, :, :] +
                a2 * s[:, 2:3, :, :]
        )

        # 输出 gate
        out = torch.sigmoid(self.proj(out))
        return out


class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )

        mask = torch.ones_like(self.weight)
        kh, kw = self.weight.shape[-2:]
        mask[:, :, kh // 2, kw // 2] = 0.0  # 中心置0
        self.register_buffer("mask", mask)

    def forward(self, x):
        weight = self.weight * self.mask
        return F.conv2d(
            x, weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )

    class AConvLKS(nn.Module):
        """
        AConv + lightweight LKS
        结构：
            AvgPool(2,1) -> Conv(3x3,s=2) -> DW Dilated Conv(3x3,d=d)
        说明：
            1. 先平滑再下采样，减少信息突变
            2. 下采样后接一个轻量DW膨胀卷积，扩大感受野
            3. 参数增加很小：仅增加 9*c2 个depthwise参数
        """

        def __init__(self, c1, c2, d=2):
            super().__init__()
            self.cv1 = Conv(c1, c2, 3, 2, 1)
            self.dw = Conv(c2, c2, 3, 1, None, g=c2, d=d)  # depthwise dilated conv

        def forward(self, x):
            x = F.avg_pool2d(x, 2, 1, 0, False, True)
            x = self.cv1(x)
            x = self.dw(x)
            return x

    class ADownLKS(nn.Module):
        """
        ADown + lightweight LKS
        结构：
            AvgPool(2,1)
              -> split
                 branch1: Conv(3x3,s=2) -> DW Dilated Conv(3x3,d=d)
                 branch2: MaxPool(3,2,1) -> Conv(1x1)
              -> Concat

        说明：
            1. 保留 ADown 的双分支信息保留能力
            2. 只在卷积分支上加 LKS，控制参数和计算量
            3. 比在 concat 后整块做 LKS 更轻
        """

        def __init__(self, c1, c2, d=2):
            super().__init__()
            assert c1 % 2 == 0, f"ADownLKS expects even input channels, but got c1={c1}"
            self.c = c2 // 2

            self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
            self.dw = Conv(self.c, self.c, 3, 1, None, g=self.c, d=d)  # depthwise dilated conv
            self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

        def forward(self, x):
            x = F.avg_pool2d(x, 2, 1, 0, False, True)
            x1, x2 = x.chunk(2, 1)

            y1 = self.cv1(x1)
            y1 = self.dw(y1)

            y2 = F.max_pool2d(x2, 3, 2, 1)
            y2 = self.cv2(y2)

            return torch.cat((y1, y2), 1)


class CGMDown(nn.Module):
    """
    Content-Guided Multi-branch Downsampling
    结构：
        1) 1x1压缩
        2) 三分支：
           - low-frequency branch
           - detail-enhanced branch
           - context branch
        3) content-guided selector
        4) 1x1 fuse + shortcut
        5) CGA refine
    """

    def __init__(self, c1, c2, k=3, s=2, e=0.5, use_cga=True):
        super().__init__()
        assert s == 2, "CGMDown主要用于stride=2下采样"

        c_ = max(int(c2 * e), 24)
        c_ = max((c_ // 3) * 3, 3)  # 保证能均分3个分支
        cb = c_ // 3
        hidden = max(c_ // 4, 16)

        # 先压缩
        self.pre = Conv(c1, c_, 1, 1)

        # branch 1: low-frequency / anti-alias
        self.pool = nn.AvgPool2d(2, 2)
        self.low_pw = Conv(cb, cb, 1, 1)

        # branch 2: detail-enhanced
        self.blur = nn.AvgPool2d(3, 1, 1)
        self.det_dw = nn.Conv2d(cb, cb, 3, 2, 1, groups=cb, bias=False)
        self.det_bn = nn.BatchNorm2d(cb)
        self.det_pw = Conv(cb, cb, 1, 1)

        # branch 3: context
        self.ctx_dw = nn.Conv2d(cb, cb, 5, 2, 2, groups=cb, bias=False)
        self.ctx_bn = nn.BatchNorm2d(cb)
        self.ctx_pw = Conv(cb, cb, 1, 1)

        # content-guided selector
        self.fc1 = nn.Conv2d(2 * c_, hidden, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden, 3, 1, bias=True)

        # fuse + shortcut
        self.fuse = Conv(c_, c2, 1, 1)
        self.short = Conv(c1, c2, 1, 2)

        self.use_cga = use_cga
        if use_cga:
            self.cga = CGA(c2, reduce_to=16, k=5)

    def forward(self, x):
        x0 = self.pre(x)  # [B, c_, H, W]
        xa, xb, xc = torch.chunk(x0, 3, dim=1)

        # 1) low-frequency branch
        low = self.low_pw(self.pool(xa))

        # 2) detail-enhanced branch
        detail_in = xb + (xb - self.blur(xb))  # 高频增强
        det = self.det_dw(detail_in)
        det = self.det_bn(det)
        det = self.det_pw(det)

        # 3) context branch
        ctx = self.ctx_dw(xc)
        ctx = self.ctx_bn(ctx)
        ctx = self.ctx_pw(ctx)

        # content-guided selector
        m = torch.cat([low, det, ctx], dim=1)
        stat = torch.cat([
            F.adaptive_avg_pool2d(m, 1),
            F.adaptive_max_pool2d(m, 1)
        ], dim=1)
        w = self.fc2(F.relu(self.fc1(stat), inplace=True))
        w = torch.softmax(w, dim=1)  # [B, 3, 1, 1]

        out = torch.cat([
            low * w[:, 0:1],
            det * w[:, 1:2],
            ctx * w[:, 2:3]
        ], dim=1)

        out = self.fuse(out) + self.short(x)

        if self.use_cga:
            out = out * self.cga(out) + out

        return out


class CGA(nn.Module):
    """
    DEA-Net: Content-Guided Attention
    输入 X:(B,C,H,W) -> 输出 W:(B,C,H,W)（每通道一张SIM）
    """

    def __init__(self, c, reduce_to=16, k=7):
        super().__init__()
        mid = min(reduce_to, c)

        # channel attention: GAP -> 1x1 -> ReLU -> 1x1
        self.ca1 = nn.Conv2d(c, mid, 1, bias=True)
        self.ca2 = nn.Conv2d(mid, c, 1, bias=True)

        # spatial attention: [avg,max] -> 7x7 conv
        self.sa = nn.Conv2d(2, 1, k, padding=k // 2, bias=True)

        # refine: concat(X, Wcoa) -> shuffle -> 7x7 group conv(groups=C)
        self.refine = nn.Conv2d(2 * c, c, k, padding=k // 2, groups=c, bias=True)

    def forward(self, x):
        # Wc
        xc = F.adaptive_avg_pool2d(x, 1)
        wc = self.ca2(F.relu(self.ca1(xc), inplace=True))  # (B,C,1,1)

        # Ws
        xs_avg = torch.mean(x, dim=1, keepdim=True)
        xs_max = torch.max(x, dim=1, keepdim=True)[0]
        ws = self.sa(torch.cat([xs_avg, xs_max], dim=1))  # (B,1,H,W)

        # coarse SIM
        wcoa = wc + ws  # (B,C,H,W) broadcast

        # refine
        z = torch.cat([x, wcoa], dim=1)  # (B,2C,H,W)
        z = channel_shuffle(z, groups=2)
        w = torch.sigmoid(self.refine(z))  # (B,C,H,W)
        return w


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.shape
    assert c % groups == 0
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class HaarDWT(nn.Module):
    """
    轻量 Haar 小波分解。
    输入:  x  [B, C, H, W]
    输出:  LL, LH, HL, HH  [B, C, H/2, W/2]
    """

    def forward(self, x):
        _, _, h, w = x.shape

        # 防止奇数尺寸报错
        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(x, (0, w % 2, 0, h % 2))

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        # Haar 2D DWT
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 + x01 - x10 - x11) * 0.5
        hl = (x00 - x01 + x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        return ll, lh, hl, hh


class CGWTDown(nn.Module):
    """
    Content-Guided Wavelet Downsampling

    用来替换 YOLO 中的 stride=2 Conv：
        Conv(c1, c2, k=3, s=2)
    替换为：
        CGWTDown(c1, c2, k=3, s=2)

    核心思想：
    1. DWT 直接完成无学习参数的频域下采样；
    2. LL 分支保留低频结构和轮廓；
    3. LH/HL/HH 分支保留高频边缘和小目标细节；
    4. 内容引导门控自适应融合低频 / 高频；
    5. 保留普通 stride=2 卷积分支，提高稳定性。
    """

    def __init__(
            self,
            c1,
            c2,
            k=3,
            s=2,
            p=None,
            g=1,
            d=1,
            act=True,
            mid_ratio=0.5,
            use_shortcut=True
    ):
        super().__init__()

        assert s == 2, "CGWTDown 主要用于替换 stride=2 的下采样卷积"

        mid = max(16, int(c2 * mid_ratio))
        mid = min(mid, c2)

        self.dwt = HaarDWT()

        # 低频结构分支：LL 主要对应目标整体轮廓、背景平滑结构
        self.low_path = nn.Sequential(
            Conv(c1, mid, 1, 1, act=act),
            Conv(mid, mid, 5, 1, g=mid, act=act)
        )

        # 高频方向分支：LH / HL / HH 先进行轻量方向融合
        # groups=c1 表示每个原始通道内部的三个高频子带独立融合，参数量极小
        self.hf_fuse = nn.Sequential(
            nn.Conv2d(c1 * 3, c1, kernel_size=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU() if act is True else nn.Identity()
        )

        self.high_path = nn.Sequential(
            Conv(c1, mid, 1, 1, act=act),
            Conv(mid, mid, k, 1, g=mid, act=act)
        )

        # 内容引导选择器：根据当前位置，自适应选择低频结构或高频细节
        self.selector = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU() if act is True else nn.Identity(),
            nn.Conv2d(8, 2, kernel_size=1, bias=True)
        )

        # 普通空间下采样分支：防止纯频域分支训练不稳定
        self.spatial_path = Conv(c1, mid, k, s, p, g, d, act=act)

        # 融合输出
        self.out_proj = Conv(mid * 2, c2, 1, 1, act=False)

        # 下采样残差分支，提高训练稳定性
        self.shortcut = Conv(c1, c2, 1, 2, act=False) if use_shortcut else None

        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # 1. 小波下采样
        ll, lh, hl, hh = self.dwt(x)

        # 2. 低频结构分支
        low = self.low_path(ll)

        # 3. 高频细节分支
        high_raw = torch.cat([lh, hl, hh], dim=1)
        high_raw = self.hf_fuse(high_raw)
        high = self.high_path(high_raw)

        # 4. 内容引导低频 / 高频融合
        mix = torch.cat([low, high], dim=1)
        avg = torch.mean(mix, dim=1, keepdim=True)
        mx = torch.amax(mix, dim=1, keepdim=True)

        weight = self.selector(torch.cat([avg, mx], dim=1))
        weight = torch.softmax(weight, dim=1)

        freq = low * weight[:, 0:1, :, :] + high * weight[:, 1:2, :, :]

        # 5. 普通空间下采样分支
        spatial = self.spatial_path(x)

        # 6. 融合输出
        out = self.out_proj(torch.cat([spatial, freq], dim=1))

        if self.shortcut is not None:
            out = out + self.shortcut(x)

        return self.act(out)


class CED(nn.Module):
    """Compact enhancement-downsampling block with a Conv-compatible interface."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, e=0.5):
        """Initialize CED so it can be swapped with Conv directly from model YAML."""
        super().__init__()
        if s not in {1, 2}:
            raise ValueError(f"CED only supports stride 1 or 2, but got s={s}.")

        c_ = max(int(c2 * e), 1)
        hidden_act = self.default_act if act is True else nn.Identity()
        self.s = s
        self.cv1 = Conv(c1, c_, 1, 1, act=hidden_act)
        self.dwconv = Conv(c_, c_, k, 1, p, g=c_, d=d, act=hidden_act)
        self.cv2 = Conv(c_ * 4 if s == 2 else c_, c2, 1, 1, act=act)

    def forward(self, x):
        """Apply local enhancement first, then optional focus-style downsampling."""
        x = self.dwconv(self.cv1(x))
        if self.s == 2:
            h, w = x.shape[-2:]
            if h % 2 != 0 or w % 2 != 0:
                x = F.pad(x, (0, w % 2, 0, h % 2))
            x = torch.cat(
                (
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ),
                dim=1,
            )
        return self.cv2(x)


class ECALayer(nn.Module):
    """
    Efficient Channel Attention.
    只引入极少参数，用于通道重标定。
    """
    def __init__(self, c, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                     # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)      # [B, 1, C]
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)    # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y


class SpatialBranchGate(nn.Module):
    """
    Spatial softmax gate for multi-branch selection.
    根据多分支特征的 avg/max 空间统计，生成每个位置的分支权重。
    """
    def __init__(self, n_branch=3, k=7):
        super().__init__()
        self.gate = nn.Conv2d(2, n_branch, kernel_size=k, padding=k // 2, bias=True)

    def forward(self, feats):
        # feats: list of [B, C, H, W]
        x = torch.cat(feats, dim=1)              # [B, n*C, H, W]
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)          # [B, 2, H, W]
        w = self.gate(s)                         # [B, n_branch, H, W]
        w = torch.softmax(w, dim=1)
        return w


class LiteChannelGate(nn.Module):
    """
    Muon-friendly lightweight channel gate.
    用 Conv2d 替代 Conv1d，避免 3D 参数导致 Muon 报错。
    """
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(c, c, 1, groups=c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.gate(self.pool(x))
        return x * w


class DCED(nn.Module):
    """
    DGF-CED: Detail-Guided Focus Compact Enhancement Downsampling.

    Compared with original CED:
    1. Uses three lightweight enhancement branches:
       - local detail branch
       - dilated context branch
       - high-frequency branch
    2. Uses spatial softmax gate for dynamic branch selection.
    3. Uses ECA for lightweight channel recalibration.
    4. Adds anti-alias smoothing before focus-style downsampling.

    It keeps a Conv-compatible interface and can replace Conv/CED in YAML.
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, e=0.5):
        super().__init__()

        if s not in {1, 2}:
            raise ValueError(f"DGFCED only supports stride 1 or 2, but got s={s}.")

        c_ = max(int(c2 * e), 1)
        hidden_act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.s = s

        # 1. channel compression
        self.cv1 = Conv(c1, c_, 1, 1, act=hidden_act)

        # 2. local detail branch
        self.local_branch = Conv(
            c_, c_,
            k=k, s=1, p=p,
            g=c_, d=d,
            act=hidden_act
        )

        # 3. dilated context branch
        self.context_branch = Conv(
            c_, c_,
            k=3, s=1, p=2,
            g=c_, d=2,
            act=hidden_act
        )

        # 4. high-frequency branch
        self.hf_branch = Conv(
            c_, c_,
            k=3, s=1, p=1,
            g=c_,
            act=hidden_act
        )

        # 5. dynamic spatial branch selection
        self.branch_gate = SpatialBranchGate(n_branch=3, k=7)

        # 6. lightweight channel recalibration
        self.eca = LiteChannelGate(c_)

        # 8. output projection
        self.cv2 = Conv(c_ * 4 if s == 2 else c_, c2, 1, 1, act=act)

    def forward(self, x):
        x = self.cv1(x)

        # local texture/detail
        f_local = self.local_branch(x)

        # dilated context
        f_context = self.context_branch(x)

        # high-frequency detail: x - low-frequency(x)
        x_low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_high = x - x_low
        f_high = self.hf_branch(x_high)

        # dynamic branch selection
        w = self.branch_gate([f_local, f_context, f_high])
        x = (
            f_local * w[:, 0:1, :, :] +
            f_context * w[:, 1:2, :, :] +
            f_high * w[:, 2:3, :, :]
        )

        # residual compensation + channel recalibration
        x = self.eca(x)

        # optional focus-style downsampling
        if self.s == 2:

            h, w_ = x.shape[-2:]
            if h % 2 != 0 or w_ % 2 != 0:
                x = F.pad(x, (0, w_ % 2, 0, h % 2))

            x = torch.cat(
                (
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ),
                dim=1,
            )

        return self.cv2(x)




class SFDDown(nn.Module):
    """
    Small-object frequency-preserving downsampling.

    This module avoids using a single stride-2 convolution for UAV/small-object
    features. It fuses anti-aliased low-frequency features, lossless
    space-to-depth samples, and Haar high-frequency details with a lightweight
    content gate.
    """

    def __init__(self, c1, c2, k=3, s=2, e=0.5, use_gate=True):
        super().__init__()
        assert s == 2, "SFDDown is designed for stride-2 downsampling"

        c_mid = max(int(c2 * e), 24)
        self.branch_c = max(c_mid // 3, 8)
        self.use_gate = use_gate

        self.low_proj = Conv(c1, self.branch_c, 1, 1)
        self.low_dw = Conv(self.branch_c, self.branch_c, k, 1, g=self.branch_c)

        self.spd_proj = Conv(4 * c1, self.branch_c, 1, 1)
        self.spd_dw = Conv(self.branch_c, self.branch_c, k, 1, g=self.branch_c)

        self.high_proj = Conv(3 * c1, self.branch_c, 1, 1)
        self.high_dw = Conv(self.branch_c, self.branch_c, k, 1, g=self.branch_c)

        if use_gate:
            hidden = max(self.branch_c, 16)
            self.gate = nn.Sequential(
                nn.Conv2d(6 * self.branch_c, hidden, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 3 * self.branch_c, 1, bias=True),
            )

        self.fuse = Conv(3 * self.branch_c, c2, 1, 1)
        self.short = Conv(c1, c2, 1, 1, act=False)

        blur = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ) / 16.0
        self.register_buffer("blur_kernel", blur.view(1, 1, 3, 3), persistent=False)

    @staticmethod
    def _pad_even(x):
        _, _, h, w = x.shape
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x

    def _blur_down(self, x):
        x = self._pad_even(x)
        c = x.shape[1]
        weight = self.blur_kernel.to(device=x.device, dtype=x.dtype).expand(c, 1, 3, 3)
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, weight, stride=2, groups=c)

    def _haar_high(self, x):
        x = self._pad_even(x)
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return torch.cat((lh, hl, hh), dim=1)

    def forward(self, x):
        low = self.low_dw(self.low_proj(self._blur_down(x)))
        spd = self.spd_dw(self.spd_proj(F.pixel_unshuffle(self._pad_even(x), 2)))
        high = self.high_dw(self.high_proj(self._haar_high(x)))

        if self.use_gate:
            m = torch.cat((low, spd, high), dim=1)
            stat = torch.cat(
                (F.adaptive_avg_pool2d(m, 1), F.adaptive_max_pool2d(m, 1)),
                dim=1,
            )
            gate = self.gate(stat).view(x.shape[0], 3, self.branch_c, 1, 1)
            gate = torch.softmax(gate, dim=1)
            out = torch.cat(
                (low * gate[:, 0], spd * gate[:, 1], high * gate[:, 2]),
                dim=1,
            )
        else:
            out = torch.cat((low, spd, high), dim=1)

        return self.fuse(out) + self.short(self._blur_down(x))
