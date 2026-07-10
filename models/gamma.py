import torch
import torch.nn as nn


class AdaptiveGammaNet(nn.Module):
    """
    Learn image-dependent gamma correction.
    """

    def __init__(self, channels=1):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.gamma_head = nn.Sequential(

            nn.Conv2d(channels,16,1),

            nn.ReLU(inplace=True),

            nn.Conv2d(16,8,1),

            nn.ReLU(inplace=True),

            nn.Conv2d(8,1,1),

            nn.Sigmoid()

        )

    def forward(self, G):

        feat = self.pool(G)

        gamma = self.gamma_head(feat)

        gamma = 0.35 + gamma * 0.35

        return gamma
