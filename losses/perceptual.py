import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """VGG Perceptual Loss with proper device handling"""
    
    def __init__(self, device=None):
        super().__init__()
        # Load VGG and move to device
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Move to device (GPU if available)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vgg = vgg.to(device)
        self.device = device
        
    def forward(self, x, y):
        # Ensure inputs and model are on the same device
        if x.device != self.device:
            x = x.to(self.device)
        if y.device != self.device:
            y = y.to(self.device)
            
        # Handle single-channel inputs (repeat to 3 channels)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        
        # Move model to the same device as input (just in case)
        if next(self.vgg.parameters()).device != x.device:
            self.vgg = self.vgg.to(x.device)
            self.device = x.device
        
        with torch.no_grad():
            feat_x = self.vgg(x)
            feat_y = self.vgg(y)
            
        return torch.mean((feat_x - feat_y) ** 2)
