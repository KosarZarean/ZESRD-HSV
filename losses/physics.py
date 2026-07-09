import torch
import torch.nn as nn

class PhysicsLosses(nn.Module):
    """Physical losses from ZESRD paper"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def reconstruction_loss(self, J, v):
        return torch.mean(torch.abs(J - v))
    
    def retinex_consistency(self, R, R1):
        return self.mse(R, R1)
    
    def scattering_consistency(self, t, t1, A, A1):
        return self.mse(t, t1) + self.mse(A, A1)
    
    def structure_aware_smoothness(self, L, R, gamma=10.0):
        L_dx = torch.abs(L[:, :, :, 1:] - L[:, :, :, :-1])
        L_dy = torch.abs(L[:, :, 1:, :] - L[:, :, :-1, :])
        
        R_gray = R.mean(dim=1, keepdim=True)
        R_dx = torch.abs(R_gray[:, :, :, 1:] - R_gray[:, :, :, :-1])
        R_dy = torch.abs(R_gray[:, :, 1:, :] - R_gray[:, :, :-1, :])
        
        weight_x = torch.exp(-gamma * R_dx)
        weight_y = torch.exp(-gamma * R_dy)
        
        return torch.mean(L_dx * weight_x) + torch.mean(L_dy * weight_y)
    
    def physical_prior(self, t, A, L):
        return torch.mean(torch.abs(t * A - L))
    
    def total_variation(self, x):
        return torch.mean((x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2) + \
               torch.mean((x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2)
