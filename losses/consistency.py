import torch


def consistency_loss(R, R1, G, G1):
    """
    Multi-iteration consistency loss.
    Ensures that Retinex decomposition is consistent across iterations.
    
    Args:
        R: Initial reflectance estimate [B, 1, H, W]
        R1: Second-iteration reflectance estimate [B, 1, H, W]
        G: Initial illumination estimate [B, 1, H, W]
        G1: Second-iteration illumination estimate [B, 1, H, W]
    
    Returns:
        Scalar loss value
    """
    return (
        torch.mean(torch.abs(R - R1)) +
        torch.mean(torch.abs(G - G1))
    )
