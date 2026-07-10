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

    def forward(self, image: torch.Tensor, outputs: dict) -> torch.Tensor:
        """
        ورودی:
            image: [B, 3, H, W]
            outputs: شامل کلیدهای ["enhanced", "R", "G", "R1", "G1"]
        """
        # استخراج مقادیر از دیکشنری
        enhanced = outputs["enhanced"]
        R = outputs["R"]
        G = outputs["G"]
        R1 = outputs["R1"]
        G1 = outputs["G1"]

        # محاسبه ضرایب
        Lrec = physics_loss(image, R, G)
        Ltv = smoothness_loss(G, R)
        Lcons = consistency_loss(R, R1, G, G1)
        Lvgg = self.vgg(enhanced, image)
        Ledge = self.edge(enhanced, image)

        # ترکیب نهایی با وزن‌های ثابت
        total = (
            6.0 * Lrec +
            1.0 * Ltv +
            1.5 * Lcons +
            0.2 * Lvgg +
            0.3 * Ledge
        )

        return total
        
