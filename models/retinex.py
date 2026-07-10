import torch
import torch.nn as nn

from .blocks import (
    DepthwiseSeparableConv,
    ResidualDWBlock,
    MultiScaleBlock,
    CBAM
)


class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.stage1 = nn.Sequential(

            MultiScaleBlock(1,32),

            CBAM(32),

            ResidualDWBlock(32)

        )

        self.down1 = nn.Sequential(

            nn.MaxPool2d(2),

            DepthwiseSeparableConv(32,64)

        )

        self.stage2 = nn.Sequential(

            ResidualDWBlock(64),

            CBAM(64),

            ResidualDWBlock(64)

        )

        self.down2 = nn.Sequential(

            nn.MaxPool2d(2),

            DepthwiseSeparableConv(64,128)

        )

        self.stage3 = nn.Sequential(

            ResidualDWBlock(128),

            CBAM(128),

            ResidualDWBlock(128)

        )

    def forward(self,x):

        f1=self.stage1(x)

        f2=self.stage2(self.down1(f1))

        f3=self.stage3(self.down2(f2))

        return f1,f2,f3


class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.up2=nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec2=nn.Sequential(

            ResidualDWBlock(64),

            CBAM(64)

        )

        self.up1=nn.ConvTranspose2d(

            64,
            32,
            2,
            stride=2

        )

        self.dec1=nn.Sequential(

            ResidualDWBlock(32),

            CBAM(32)

        )

    def forward(self,f1,f2,f3):

        x=self.up2(f3)

        x=x+f2

        x=self.dec2(x)

        x=self.up1(x)

        x=x+f1

        x=self.dec1(x)

        return x


class RetinexNet(nn.Module):

    """
    Estimate

    Reflectance R

    Illumination G

    """

    def __init__(self):

        super().__init__()

        self.encoder=Encoder()

        self.decoder=Decoder()

        self.reflectance_head=nn.Sequential(

            DepthwiseSeparableConv(32,16),

            nn.Conv2d(16,1,3,padding=1),

            nn.Sigmoid()

        )

        self.illumination_head=nn.Sequential(

            DepthwiseSeparableConv(32,16),

            nn.Conv2d(16,1,3,padding=1),

            nn.Sigmoid()

        )

    def forward(self,V):

        f1,f2,f3=self.encoder(V)

        feat=self.decoder(f1,f2,f3)

        R=self.reflectance_head(feat)

        G=self.illumination_head(feat)

        return R,G
