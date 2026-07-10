import torch


def smoothness_loss(G, R):
    """
    Total variation smoothness loss with edge-aware weighting.
    Encourages smooth illumination and reflectance while preserving edges.
    
    Args:
        G: Illumination map [B, 1, H, W]
        R: Reflectance map [B, 1, H, W]
    
    Returns:
        Scalar loss value
    """
    dx = torch.abs(G[:, :, :, 1:] - G[:, :, :, :-1])
    dy = torch.abs(G[:, :, 1:, :] - G[:, :, :-1, :])
    
    rx = torch.abs(R[:, :, :, 1:] - R[:, :, :, :-1])
    ry = torch.abs(R[:, :, 1:, :] - R[:, :, :-1, :])
    
    return (
        (dx * torch.exp(-10 * rx)).mean() +
        (dy * torch.exp(-10 * ry)).mean()
    )
