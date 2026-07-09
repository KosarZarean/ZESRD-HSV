import torch
import torch.nn as nn
from .physics import PhysicsLosses
from .perceptual import VGGPerceptualLoss

class TotalLoss(nn.Module):
    """Complete loss function from ZESRD paper"""
    
    def __init__(self, config):
        super().__init__()
        self.physics = PhysicsLosses()
        # ✅ Pass device to VGG loss
        self.perceptual = VGGPerceptualLoss(device=config.DEVICE)
        self.config = config
        
    def forward(self, v, outputs):
        # ... (بقیه کد保持不变) ...
        R, L = outputs['R'], outputs['L']
        t, A = outputs['t'], outputs['A']
        J_ret = outputs['J_ret']
        J_scat = outputs['J_scat']
        R1, L1 = outputs['R1'], outputs['L1']
        t1, A1 = outputs['t1'], outputs['A1']
        
        # Reconstruction Loss
        loss_rec = self.physics.reconstruction_loss(J_ret, v) + \
                   self.physics.reconstruction_loss(J_scat, v)
        
        # Multi-stage Consistency
        loss_R_cons = self.config.LAMBDA_R_CONS * self.physics.retinex_consistency(R, R1)
        loss_v_cons = self.config.LAMBDA_V_CONS * self.physics.scattering_consistency(t, t1, A, A1)
        
        # Structure-aware Smoothness
        loss_ist = self.config.LAMBDA_IST * self.physics.structure_aware_smoothness(L, R)
        
        # Physical Prior
        loss_prior = self.config.LAMBDA_PRIOR * self.physics.physical_prior(t, A, L)
        
        # VGG Perceptual (✅ now works with correct device)
        loss_vgg = self.config.LAMBDA_VGG * self.perceptual(J_ret, v)
        
        # Total
        total_loss = (
            self.config.LAMBDA_REC * loss_rec +
            loss_R_cons +
            loss_v_cons +
            loss_ist +
            loss_prior +
            loss_vgg
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'rec': loss_rec.item(),
            'R_cons': loss_R_cons.item(),
            'v_cons': loss_v_cons.item(),
            'ist': loss_ist.item(),
            'prior': loss_prior.item(),
            'vgg': loss_vgg.item()
        }
        
        return total_loss, loss_dict
