import torch

def physics_loss(I, R, G):
    reconstruction = R * G
    return torch.mean(torch.abs(reconstruction - I))
