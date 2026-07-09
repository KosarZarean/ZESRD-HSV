import torch
import torch.nn as nn
from .blocks import DepthwiseSeparableConv, ResidualDWBlock, SpatialAttention

class ScatteringNetwork(nn.Module):
    """Scattering Decomposition Network: V → t, A"""
    def __init__(self, base_channels=32):
        super().__init__()
        
        self.conv1 = DepthwiseSeparableConv(1, base_channels)
        self.conv2 = DepthwiseSeparableConv(base_channels, base_channels)
        self.res1 = ResidualDWBlock(base_channels)
        self.res2 = ResidualDWBlock(base_channels)
        self.att = SpatialAttention()
        
        self.t_head = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.A_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, v):
        x = self.conv1(v)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.att(x)
        
        t = self.t_head(x)
        A = self.A_head(x)
        
        # Physical constraint
        t = torch.clamp(t, 0.1, 0.9)
        
        return t, A
