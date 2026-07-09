# ============================================================
# Cell: Final Run - Fixed Version
# ============================================================

import sys
import torch
import importlib

# 1. Clear all caches
print("🔄 Clearing caches...")
torch.cuda.empty_cache()
import gc
gc.collect()

# 2. Force reload modules
sys.path.append('/content/ZESRD-HSV')

if 'losses.perceptual' in sys.modules:
    del sys.modules['losses.perceptual']
if 'losses.total_loss' in sys.modules:
    del sys.modules['losses.total_loss']
if 'train' in sys.modules:
    del sys.modules['train']
if 'evaluate' in sys.modules:
    del sys.modules['evaluate']

print("✅ Caches cleared and modules reloaded!")

# 3. Fix the perceptual loss file permanently
print("🔧 Applying permanent fix to perceptual.py...")

%%writefile /content/ZESRD-HSV/losses/perceptual.py
import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """VGG Perceptual Loss - Fixed device handling"""
    
    def __init__(self, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Load VGG and immediately move to device
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        
        print(f"✅ VGG perceptual loss initialized on {device}")
        
    def forward(self, x, y):
        # Ensure inputs are on the correct device
        x = x.to(self.device)
        y = y.to(self.device)
            
        # Handle single-channel inputs
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            feat_x = self.vgg(x)
            feat_y = self.vgg(y)
            
        return torch.mean((feat_x - feat_y) ** 2)

print("✅ perceptual.py fixed!")

# 4. Fix total_loss.py
print("🔧 Applying permanent fix to total_loss.py...")

%%writefile /content/ZESRD-HSV/losses/total_loss.py
import torch
import torch.nn as nn
from .physics import PhysicsLosses
from .perceptual import VGGPerceptualLoss

class TotalLoss(nn.Module):
    """Complete loss function with proper device handling"""
    
    def __init__(self, config):
        super().__init__()
        self.physics = PhysicsLosses()
        # ✅ Pass device to VGG loss
        self.perceptual = VGGPerceptualLoss(device=config.DEVICE)
        self.config = config
        
    def forward(self, v, outputs):
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
        
        # VGG Perceptual - Now works correctly
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

print("✅ total_loss.py fixed!")

# 5. Import and run
print("\n" + "="*70)
print("🚀 Running Evaluation (Fixed Version)")
print("="*70)

from config import Config
from evaluate import evaluate_zesrd

print(f"Device: {Config.DEVICE}")
print(f"Iterations: {Config.TOTAL_ITERATIONS}")
print("-"*70)

# Run evaluation
psnrs, ssims = evaluate_zesrd()

# Display results
print("\n" + "="*50)
print("📊 FINAL RESULTS")
print("="*50)
print(f"✅ Mean PSNR: {sum(psnrs)/len(psnrs):.2f} dB")
print(f"✅ Mean SSIM: {sum(ssims)/len(ssims):.4f}")
print("="*50)
