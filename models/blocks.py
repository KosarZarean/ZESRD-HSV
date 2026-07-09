# Cell 1: Create models/blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding, 
            groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.pointwise(self.depthwise(x)))


class ResidualDWBlock(nn.Module):
    """Residual Block with DW-Conv"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y


class MultiScaleFeatureFusion(nn.Module):
    """Multi-scale Feature Extraction (3x3, 5x5, 7x7)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch3 = DepthwiseSeparableConv(in_ch, out_ch, 3, 1)
        self.branch5 = DepthwiseSeparableConv(in_ch, out_ch, 5, 2)
        self.branch7 = DepthwiseSeparableConv(in_ch, out_ch, 7, 3)
        self.fuse = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out3 = self.branch3(x)
        out5 = self.branch5(x)
        out7 = self.branch7(x)
        combined = torch.cat([out3, out5, out7], dim=1)
        return self.act(self.fuse(combined))
