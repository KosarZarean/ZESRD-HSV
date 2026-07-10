import torch
import torch.nn as nn
from losses.perceptual import VGGPerceptualLoss
from losses.edge import EdgeLoss
from losses.physics import physics_loss
from losses.smoothness import smoothness_loss
from losses.consistency import consistency_loss


class TotalLoss(nn.Module):
    """
    Composite loss function for ZESRD-HSV zero-shot image enhancement.
    
    Interface Contract:
        - Accepts: forward(image: Tensor[B,3,H,W], outputs: dict)
        - outputs dict MUST contain keys: ['enhanced', 'R', 'G', 'R1', 'G1']
        - All tensors must be on the same device
    
    Loss Components:
        - Lrec: Physics-based reconstruction loss (weight: 6.0)
        - Ltv: Total variation smoothness loss (weight: 1.0)
        - Lcons: Multi-iteration consistency loss (weight: 1.5)
        - Lvgg: VGG perceptual loss (weight: 0.2)
        - Ledge: Edge preservation loss (weight: 0.3)
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.vgg = VGGPerceptualLoss()
        self.edge = EdgeLoss()
        
        # Loss weight coefficients
        self.weights = {
            'reconstruction': 6.0,
            'smoothness': 1.0,
            'consistency': 1.5,
            'perceptual': 0.2,
            'edge': 0.3,
        }
    
    def forward(self, image: torch.Tensor, outputs: dict) -> torch.Tensor:
        """
        Compute total loss from image and model outputs.
        
        Args:
            image: Input image tensor [B, 3, H, W], assumed normalized to [0, 1]
            outputs: Dictionary with keys:
                - 'enhanced': Enhanced RGB image [B, 3, H, W]
                - 'R': Reflectance map [B, 1, H, W]
                - 'G': Illumination map [B, 1, H, W]
                - 'R1': Second-iteration reflectance [B, 1, H, W]
                - 'G1': Second-iteration illumination [B, 1, H, W]
        
        Returns:
            Scalar loss value
        
        Raises:
            KeyError: If required keys are missing from outputs dict
            AssertionError: If tensor shapes are incompatible
        """
        # Validate and extract outputs
        required_keys = ['enhanced', 'R', 'G', 'R1', 'G1']
        for key in required_keys:
            if key not in outputs:
                raise KeyError(
                    f"Missing required key '{key}' in outputs dict. "
                    f"Got keys: {list(outputs.keys())}"
                )
        
        enhanced = outputs['enhanced']
        R = outputs['R']
        G = outputs['G']
        R1 = outputs['R1']
        G1 = outputs['G1']
        
        # Validate tensor shapes and devices
        self._validate_tensors(image, enhanced, R, G, R1, G1)
        
        # Convert image to appropriate channels if needed
        # For reconstruction loss, we need to work with the V channel
        # but the physics loss expects the full image representation
        image_for_reconstruction = image.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Compute individual loss components
        try:
            Lrec = physics_loss(image_for_reconstruction, R, G)
            Ltv = smoothness_loss(G, R)
            Lcons = consistency_loss(R, R1, G, G1)
            Lvgg = self.vgg(enhanced, image)
            Ledge = self.edge(enhanced, image)
        except RuntimeError as e:
            raise RuntimeError(
                f"Error computing loss components: {str(e)}. "
                f"Check tensor shapes and device placement."
            )
        
        # Check for NaN values
        loss_components = {'Lrec': Lrec, 'Ltv': Ltv, 'Lcons': Lcons, 'Lvgg': Lvgg, 'Ledge': Ledge}
        for name, loss in loss_components.items():
            if torch.isnan(loss):
                raise ValueError(
                    f"NaN detected in {name}. This may indicate:"
                    f"\n  1. Numerical instability in computation"
                    f"\n  2. Invalid input values (check for NaN in inputs)"
                    f"\n  3. Division by zero or log of zero"
                )
        
        # Weighted combination
        total = (
            self.weights['reconstruction'] * Lrec +
            self.weights['smoothness'] * Ltv +
            self.weights['consistency'] * Lcons +
            self.weights['perceptual'] * Lvgg +
            self.weights['edge'] * Ledge
        )
        
        return total
    
    def _validate_tensors(
        self,
        image: torch.Tensor,
        enhanced: torch.Tensor,
        R: torch.Tensor,
        G: torch.Tensor,
        R1: torch.Tensor,
        G1: torch.Tensor,
    ) -> None:
        """
        Validate tensor shapes and device placement.
        
        Args:
            image: [B, 3, H, W]
            enhanced: [B, 3, H, W]
            R, G, R1, G1: [B, 1, H, W]
        
        Raises:
            AssertionError: If shapes or devices don't match
        """
        batch_size, _, height, width = image.shape
        device = image.device
        
        # Check image shape
        assert image.ndim == 4, f"image must be 4D, got {image.ndim}D"
        assert image.shape[1] == 3, f"image channels must be 3, got {image.shape[1]}"
        
        # Check enhanced shape
        assert enhanced.ndim == 4, f"enhanced must be 4D, got {enhanced.ndim}D"
        assert enhanced.shape[1] == 3, f"enhanced channels must be 3, got {enhanced.shape[1]}"
        assert enhanced.shape == image.shape, \
            f"enhanced shape {enhanced.shape} != image shape {image.shape}"
        
        # Check decomposition maps shape
        for tensor, name in [(R, 'R'), (G, 'G'), (R1, 'R1'), (G1, 'G1')]:
            assert tensor.ndim == 4, f"{name} must be 4D, got {tensor.ndim}D"
            assert tensor.shape[0] == batch_size, \
                f"{name} batch size {tensor.shape[0]} != {batch_size}"
            assert tensor.shape[1] == 1, \
                f"{name} channels must be 1, got {tensor.shape[1]}"
            assert tensor.shape[2] == height, \
                f"{name} height {tensor.shape[2]} != {height}"
            assert tensor.shape[3] == width, \
                f"{name} width {tensor.shape[3]} != {width}"
        
        # Check device consistency
        for tensor, name in [(enhanced, 'enhanced'), (R, 'R'), (G, 'G'), (R1, 'R1'), (G1, 'G1')]:
            assert tensor.device == device, \
                f"{name} on {tensor.device}, expected {device}"
