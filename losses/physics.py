import torch


def physics_loss(I, R, G):
    """
    Physics-based reconstruction loss.
    Enforces the image formation model: I = R * G
    
    Args:
        I: Input image [B, C, H, W]
        R: Reflectance estimate [B, 1, H, W]
        G: Illumination estimate [B, 1, H, W]
    
    Returns:
        Scalar loss value (reconstruction error)
    """
    reconstruction = R * G
    return torch.mean(torch.abs(reconstruction - I))
