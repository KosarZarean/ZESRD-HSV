import torch
import torch.nn as nn
from .blocks import (
    DepthwiseSeparableConv, ResidualDWBlock,
    ChannelAttention, SpatialAttention, MultiScaleFeatureFusion
)

class RetinexNetwork(nn.Module):
    """Retinex Decomposition Network: V → R, L"""
    def __init__(self, base_channels=32):
        super().__init__()
        
        self.in_conv = nn.Conv2d(1, base_channels, 3, padding=1)
        self.ms_fusion = MultiScaleFeatureFusion(base_channels, base_channels)
        
        self.res1 = ResidualDWBlock(base_channels)
        self.res2 = ResidualDWBlock(base_channels)
        
        self.channel_att = ChannelAttention(base_channels)
        self.spatial_att = SpatialAttention()
        
        self.R_head = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.L_head = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, v):
        x = self.in_conv(v)
        x = self.ms_fusion(x)
        x = self.res1(x)
        x = self.res2(x)
        
        feat_R = self.channel_att(x)
        feat_L = self.spatial_att(x)
        
        R = self.R_head(feat_R)
        L = self.L_head(feat_L)
        
        return R, L
