import torch
import torch.nn as nn
from .retinex import RetinexNetwork
from .scattering import ScatteringNetwork

class ZESRD_HSV(nn.Module):
    """Complete ZESRD on V channel"""
    def __init__(self, base_channels=32, K=8):
        super().__init__()
        self.retinex = RetinexNetwork(base_channels)
        self.scattering = ScatteringNetwork(base_channels)
        self.K = K
        
    def forward(self, v):
        # Stage 1: Decomposition
        R, L = self.retinex(v)
        t, A = self.scattering(v)
        
        # Reconstruction
        J_ret = R * L
        J_scat = torch.clamp((v - A * (1 - t)) / (t + 1e-5), 0, 1)
        
        # Stage 2: Re-decomposition for Consistency
        R1, L1 = self.retinex(J_scat.detach())
        t1, A1 = self.scattering(L1)
        
        return {
            'R': R, 'L': L, 't': t, 'A': A,
            'J_ret': J_ret, 'J_scat': J_scat,
            'R1': R1, 'L1': L1, 't1': t1, 'A1': A1
        }
