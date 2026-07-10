import torch
import torch.nn as nn


class AdaptiveGammaNet(nn.Module):
    """
    Learn image-dependent gamma correction with numerical stability.
    
    Outputs gamma values in range [0.35, 0.70] to prevent numerical instability
    and ensure meaningful gamma correction.
    """

    def __init__(self, channels=1):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.gamma_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # Numerical stability epsilon
        self.register_buffer('eps', torch.tensor(1e-6))

    def forward(self, G):
        """
        Compute adaptive gamma correction.
        
        Args:
            G: Illumination map [B, 1, H, W]
        
        Returns:
            gamma: Gamma correction factors [B, 1, 1, 1]
        
        Ensures:
            - No NaN values in output
            - Gamma in range [0.35, 0.70]
            - Numerically stable computation
        """
        # Global average pooling
        feat = self.pool(G)  # [B, 1, 1, 1]
        
        # Compute base gamma (sigmoid output in [0, 1])
        gamma_base = self.gamma_head(feat)  # [B, 1, 1, 1]
        
        # Clamp to prevent extreme values
        gamma_base = torch.clamp(gamma_base, min=1e-6, max=1.0 - 1e-6)
        
        # Map to [0.35, 0.70] range for stable gamma correction
        # This prevents both over-brightening (gamma too small) 
        # and under-brightening (gamma too large)
        gamma = 0.35 + gamma_base * 0.35  # Range: [0.35, 0.70]
        
        # Final safety check for NaN/Inf
        if torch.isnan(gamma).any():
            print("Warning: NaN detected in gamma, replacing with default value")
            gamma = torch.full_like(gamma, 0.525)  # Midpoint of [0.35, 0.70]
        
        if torch.isinf(gamma).any():
            print("Warning: Inf detected in gamma, replacing with default value")
            gamma = torch.full_like(gamma, 0.525)
        
        return gamma
