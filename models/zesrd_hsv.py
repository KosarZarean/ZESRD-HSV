import torch
import torch.nn as nn
import kornia

from .retinex import RetinexNet
from .scattering import ScatteringNet
from .koschmieder import KoschmiederModel
from .gamma import AdaptiveGammaNet

class ZESRD_HSV(nn.Module):
    def __init__(self, base_channels=32, K=8):
        super().__init__()
        self.base_channels = base_channels
        self.K = K
        
        self.retinex = RetinexNet()
        self.scattering = ScatteringNet()
        self.koschmieder = KoschmiederModel()
        self.gamma = AdaptiveGammaNet()

    def forward(self, rgb):
        # 1. RGB -> HSV
        hsv = kornia.color.rgb_to_hsv(rgb)
        H, S, V = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]

        # 2. Retinex decomposition
        R, G = self.retinex(V)

        # 3. Scattering estimation
        A, t = self.scattering(G)

        # 4. Koschmieder inverse
        G_restore = self.koschmieder.inverse(G, A, t)

        # 5. Adaptive Gamma
        gamma = self.gamma(G_restore)
        G_restore = torch.pow(G_restore + 1e-6, gamma)

        # 6. Enhanced illumination
        V_new = torch.clamp(G_restore * R, 0, 1)

        # 7. HSV -> RGB
        hsv_new = torch.cat([H, S, V_new], dim=1)
        rgb_new = kornia.color.hsv_to_rgb(hsv_new)

        # 8. Multi-stage consistency
        R1, G1 = self.retinex(V_new)

        # قرارداد نهایی: فقط کلیدهای مورد نیاز TotalLoss
        return {
            "enhanced": rgb_new,
            "R": R,
            "G": G,
            "R1": R1,
            "G1": G1
        }
        
