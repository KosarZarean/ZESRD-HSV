import torch
import torch.nn as nn

from .perceptual import VGGPerceptualLoss
from .edge import EdgeLoss
from .physics import physics_loss
from .smoothness import smoothness_loss
from .consistency import consistency_loss


class TotalLoss(nn.Module):

    def __init__(self, config=None):

        super().__init__()

        self.config = config

        self.vgg = VGGPerceptualLoss()

        self.edge = EdgeLoss()



    def forward(
        self,
        I,
        enhanced,
        R,
        G,
        R1,
        G1
    ):


        # Physics reconstruction

        Lrec = physics_loss(
            I,
            R,
            G
        )


        # Illumination smoothness

        Ltv = smoothness_loss(
            G,
            R
        )


        # Consistency

        Lcons = consistency_loss(
            R,
            R1,
            G,
            G1
        )


        # Perceptual

        Lvgg = self.vgg(
            enhanced,
            I
        )


        # Edge

        Ledge = self.edge(
            enhanced,
            I
        )


        total = (

            6.0 * Lrec

            +

            1.0 * Ltv

            +

            1.5 * Lcons

            +

            0.2 * Lvgg

            +

            0.3 * Ledge

        )


        return total            0.2 * Lvgg +
            0.3 * Ledge
        )
        return total
        
