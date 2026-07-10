
import torch
import torch.nn as nn
import kornia

from .retinex import RetinexNet
from .scattering import ScatteringNet
from .koschmieder import KoschmiederModel
from .gamma import AdaptiveGammaNet


class ZESRD_HSV(nn.Module):

    def __init__(
        self,
        base_channels=32,
        K=8
    ):

        super().__init__()

        self.base_channels = base_channels
        self.K = K


        # =========================
        # Retinex decomposition
        # =========================

        self.retinex = RetinexNet()



        # =========================
        # Scattering estimation
        # =========================

        self.scattering = ScatteringNet()



        # =========================
        # Koschmieder physical model
        # =========================

        self.koschmieder = KoschmiederModel()



        # =========================
        # Adaptive Gamma correction
        # =========================

        self.gamma = AdaptiveGammaNet()



    def adaptive_gamma(self, x):

        gamma = (
            0.45 +
            0.20 *
            (1 - torch.mean(x))
        )

        gamma = torch.clamp(
            gamma,
            0.35,
            0.70
        )

        return torch.pow(
            x,
            gamma
        )



    def forward(self, rgb):


        # =========================
        # RGB -> HSV
        # =========================

        hsv = kornia.color.rgb_to_hsv(rgb)


        H = hsv[:,0:1]

        S = hsv[:,1:2]

        V = hsv[:,2:3]



        # =========================
        # Retinex decomposition
        # =========================

        R, G = self.retinex(V)



        # =========================
        # Scattering estimation
        # =========================

        A, t = self.scattering(G)



        # =========================
        # Koschmieder inverse
        # =========================

        G_restore = self.koschmieder.inverse(
            G,
            A,
            t
        )



        # =========================
        # Adaptive Gamma
        # =========================

        gamma = self.gamma(
            G_restore
        )


        G_restore = torch.pow(
            G_restore + 1e-6,
            gamma
        )



        # =========================
        # Enhanced illumination
        # =========================

        V_new = torch.clamp(
            G_restore * R,
            0,
            1
        )



        # =========================
        # HSV -> RGB
        # =========================

        hsv_new = torch.cat(
            [
                H,
                S,
                V_new
            ],
            dim=1
        )


        rgb_new = kornia.color.hsv_to_rgb(
            hsv_new
        )



        # =========================
        # Multi-stage consistency
        # =========================

        R1, G1 = self.retinex(
            V_new
        )



        return {

            "enhanced": rgb_new,

            "R": R,

            "G": G,

            "A": A,

            "t": t,

            "R1": R1,

            "G1": G1,

            "V": V,

            "V_new": V_new

        }
        V = hsv[:, 2:3]

        # ---------- Retinex ----------

        R, G = self.retinex(V)

        # ---------- Scattering ----------

        A, t = self.scattering(G)

        # ---------- Physics ----------

        G_restore = self.koschmieder.inverse(G, A, t)

        # ---------- Gamma ----------

        gamma = self.gamma(G_restore)

        G_restore = torch.pow(
            G_restore + 1e-6,
            gamma
        )

        # ---------- Reconstruct ----------

        V_new = torch.clamp(
            G_restore * R,
            0,
            1
        )

        hsv_new = torch.cat(

            [

                H,

                S,

                V_new

            ],

            dim=1

        )

        rgb_new = kornia.color.hsv_to_rgb(hsv_new)

        # ---------- Consistency ----------

        R1, G1 = self.retinex(V_new)

        return {

            "enhanced": rgb_new,

            "R": R,

            "G": G,

            "A": A,

            "t": t,

            "R1": R1,

            "G1": G1,

            "V": V,

            "V_new": V_new

        }
