import torch
import torch.nn as nn

from .blocks import (
    DepthwiseSeparableConv,
    ResidualDWBlock,
    CBAM,
    MultiScaleBlock
)


class ScatteringNet(nn.Module):

    """
    Estimate

    Airlight A

    Transmission t

    from illumination map G
    """

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            MultiScaleBlock(1,32),

            CBAM(32),

            ResidualDWBlock(32),

            DepthwiseSeparableConv(32,64),

            ResidualDWBlock(64),

            CBAM(64),

            DepthwiseSeparableConv(64,64),

            ResidualDWBlock(64)

        )

        self.airlight_head = nn.Sequential(

            DepthwiseSeparableConv(64,32),

            nn.Conv2d(32,1,3,padding=1),

            nn.Sigmoid()

        )

        self.transmission_head = nn.Sequential(

            DepthwiseSeparableConv(64,32),

            nn.Conv2d(32,1,3,padding=1),

            nn.Sigmoid()

        )

    def forward(self,G):

        feat = self.encoder(G)

        A = self.airlight_head(feat)

        t = self.transmission_head(feat)

        t = torch.clamp(t,0.15,0.95)

        return A,t
