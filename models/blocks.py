import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Weight Initialization
# ==========================================================

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ==========================================================
# Depthwise Separable Convolution
# ==========================================================

class DepthwiseSeparableConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):

        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        x = self.depthwise(x)

        x = self.pointwise(x)

        x = self.bn(x)

        x = self.act(x)

        return x


# ==========================================================
# Residual Depthwise Block
# ==========================================================

class ResidualDWBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = DepthwiseSeparableConv(channels, channels)

        self.conv2 = DepthwiseSeparableConv(channels, channels)

    def forward(self, x):

        return x + self.conv2(self.conv1(x))


# ==========================================================
# Channel Attention (SE Block)
# ==========================================================

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=8):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(channels,
                      channels // reduction,
                      1),

            nn.ReLU(inplace=True),

            nn.Conv2d(channels // reduction,
                      channels,
                      1),

            nn.Sigmoid()

        )

    def forward(self, x):

        weight = self.fc(self.pool(x))

        return x * weight


# ==========================================================
# Spatial Attention
# ==========================================================

class SpatialAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)

        mx, _ = torch.max(x, dim=1, keepdim=True)

        feat = torch.cat([avg, mx], dim=1)

        att = self.sigmoid(self.conv(feat))

        return x * att


# ==========================================================
# CBAM Attention
# ==========================================================

class CBAM(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel = SEBlock(channels)

        self.spatial = SpatialAttention()

    def forward(self, x):

        x = self.channel(x)

        x = self.spatial(x)

        return x


# ==========================================================
# Multi-scale Feature Block
# ==========================================================

class MultiScaleBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):

        super().__init__()

        self.conv3 = DepthwiseSeparableConv(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.conv5 = DepthwiseSeparableConv(
            in_channels,
            out_channels,
            kernel_size=5,
            padding=2
        )

        self.conv7 = DepthwiseSeparableConv(
            in_channels,
            out_channels,
            kernel_size=7,
            padding=3
        )

        self.fuse = nn.Conv2d(
            out_channels * 3,
            out_channels,
            1
        )

    def forward(self, x):

        f3 = self.conv3(x)

        f5 = self.conv5(x)

        f7 = self.conv7(x)

        feat = torch.cat([f3, f5, f7], dim=1)

        return self.fuse(feat)
