"""
Pipeline Verification Script
Comprehensive checks for ZESRD-HSV model and loss function integration.

This script:
1. Instantiates all model components
2. Runs forward pass with dummy input [1, 3, 256, 256]
3. Validates output dictionary structure
4. Computes loss without errors
5. Checks for NaN values and dimension mismatches
6. Reports comprehensive status
"""

import sys
import torch
import traceback

# Add project root to path for absolute imports
sys.path.insert(0, '.')

def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_success(msg):
    """Print success message."""
    print(f"✓ {msg}")

def print_error(msg):
    """Print error message."""
    print(f"✗ {msg}")

def print_info(msg):
    """Print info message."""
    print(f"ℹ {msg}")

def test_imports():
    """Test all critical imports."""
    print_section("1. DEPENDENCY AUDIT - Import Verification")
    
    try:
        print_info("Testing model imports...")
        from models import ZESRD_HSV
        print_success("✓ ZESRD_HSV")
        
        from models import RetinexNet, ScatteringNet, AdaptiveGammaNet
        print_success("✓ Component models (Retinex, Scattering, Gamma)")
        
        print_info("Testing loss imports...")
        from losses import TotalLoss, VGGPerceptualLoss, EdgeLoss
        print_success("✓ Loss modules (TotalLoss, VGGPerceptualLoss, EdgeLoss)")
        
        from losses import physics_loss, smoothness_loss, consistency_loss
        print_success("✓ Loss functions")
        
        print_info("Testing utils imports...")
        from utils import EMA
        print_success("✓ EMA utility")
        
        from config import Config
        print_success("✓ Config")
        
        return True
    except Exception as e:
        print_error(f"Import failed: {str(e)}")
        traceback.print_exc()\n        return False

def test_model_instantiation():
    """Test model instantiation."""
    print_section("2. MODEL INSTANTIATION")
    
    try:
        from models import ZESRD_HSV
        from config import Config
        
        device = Config.DEVICE
        print_info(f"Using device: {device}")
        
        model = ZESRD_HSV(base_channels=32, K=8)
        model = model.to(device)
        model.eval()
        print_success(f"ZESRD_HSV model instantiated and moved to {device}")
        
        return model, device
    except Exception as e:
        print_error(f"Model instantiation failed: {str(e)}")
        traceback.print_exc()
        return None, None

def test_model_forward_pass(model, device):
    """Test forward pass with dummy input."""
    print_section("3. FORWARD PASS VERIFICATION")
    
    try:
        # Create dummy input
        batch_size, channels, height, width = 1, 3, 256, 256
        dummy_input = torch.randn(batch_size, channels, height, width).to(device)
        print_info(f"Created dummy input: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print_success("Forward pass completed")
        
        # Validate output type
        if not isinstance(outputs, dict):
            raise TypeError(f"Expected dict output, got {type(outputs)}")
        print_success(f"Output is dictionary with keys: {list(outputs.keys())}")
        
        # Validate required keys
        required_keys = ['enhanced', 'R', 'G', 'R1', 'G1']
        for key in required_keys:
            if key not in outputs:
                raise KeyError(f"Missing required key: {key}")
        print_success(f"All required keys present: {required_keys}")
        
        # Validate tensor shapes and values
        print_info("Validating output tensors...")
        for key, tensor in outputs.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{key} is not a tensor: {type(tensor)}")
            
            # Check for NaN
            if torch.isnan(tensor).any():
                raise ValueError(f"{key} contains NaN values")
            
            # Check for Inf
            if torch.isinf(tensor).any():
                raise ValueError(f"{key} contains Inf values")
            
            # Check value range
            min_val, max_val = tensor.min().item(), tensor.max().item()
            print_info(f"  {key}: shape={tensor.shape}, min={min_val:.4f}, max={max_val:.4f}, device={tensor.device}")
        
        print_success("All output tensors validated")
        return outputs, dummy_input
        
    except Exception as e:
        print_error(f"Forward pass test failed: {str(e)}")
        traceback.print_exc()
        return None, None

def test_loss_computation(outputs, dummy_input, device):
    """Test loss computation."""
    print_section("4. LOSS COMPUTATION")
    
    try:
        from losses import TotalLoss
        from config import Config
        
        criterion = TotalLoss(config=Config)
        criterion = criterion.to(device)
        
        print_info("TotalLoss instantiated")
        
        # Compute loss
        loss = criterion(dummy_input, outputs)
        
        print_success("Loss computation completed")
        
        # Validate loss value
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"Loss is not a tensor: {type(loss)}")
        
        if loss.ndim != 0:
            raise ValueError(f"Loss should be scalar, got shape {loss.ndim}D")
        
        loss_value = loss.item()
        print_info(f"Loss value: {loss_value:.6f}")
        
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")
        
        if torch.isinf(loss):
            raise ValueError("Loss is Inf")
        
        if loss_value < 0:
            raise ValueError(f"Loss is negative: {loss_value}")
        
        print_success("Loss value is valid")
        return True
        
    except Exception as e:
        print_error(f"Loss computation test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_backward_pass(outputs, dummy_input, device):
    """Test backward pass."""
    print_section("5. BACKWARD PASS VERIFICATION")
    
    try:
        from losses import TotalLoss
        from config import Config
        import torch.optim as optim
        
        # Recreate model with requires_grad
        from models import ZESRD_HSV
        model = ZESRD_HSV(base_channels=32, K=8)
        model = model.to(device)
        model.train()
        
        criterion = TotalLoss(config=Config)
        criterion = criterion.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        print_info("Created new model instance with optimizer")
        
        # Forward pass
        outputs = model(dummy_input)
        loss = criterion(dummy_input, outputs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print_success("Backward pass completed successfully")
        
        # Check gradient norms
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        print_info(f"Total gradient norm: {total_norm:.6f}")
        
        if total_norm == 0:
            print_error("Gradient norm is zero - no learning signal")
            return False
        
        print_success("Gradients computed correctly")
        return True
        
    except Exception as e:
        print_error(f"Backward pass test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_device_handling(model, device):
    """Test device movement and consistency."""
    print_section("6. DEVICE HANDLING VERIFICATION")
    
    try:
        print_info(f"Testing device: {device}")
        
        # Test model on device
        for name, param in model.named_parameters():
            if param.device != device:
                raise RuntimeError(
                    f"Parameter {name} on {param.device}, expected {device}"
                )
        
        print_success(f"All model parameters on {device}")
        
        # Test buffer movement
        for name, buffer in model.named_buffers():
            if buffer.device != device:
                raise RuntimeError(
                    f"Buffer {name} on {buffer.device}, expected {device}"
                )
        
        print_success(f"All model buffers on {device}")
        
        # Test loss module device
        from losses import TotalLoss
        criterion = TotalLoss()
        criterion = criterion.to(device)
        
        for name, buffer in criterion.named_buffers():
            if buffer.device != device:
                raise RuntimeError(
                    f"Loss buffer {name} on {buffer.device}, expected {device}"
                )
        
        print_success(f"Loss module correctly moved to {device}")
        return True
        
    except Exception as e:
        print_error(f"Device handling test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print_section("ZESRD-HSV PIPELINE VERIFICATION")
    print_info("Performing comprehensive structural and runtime checks...")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print_section("VERIFICATION FAILED")
        print_error("Import tests failed. Cannot proceed.")
        return False
    
    # Test 2: Model instantiation
    model, device = test_model_instantiation()
    results['model_instantiation'] = model is not None
    if not results['model_instantiation']:
        print_section("VERIFICATION FAILED")
        print_error("Model instantiation failed. Cannot proceed.")
        return False
    
    # Test 3: Forward pass
    outputs, dummy_input = test_model_forward_pass(model, device)
    results['forward_pass'] = outputs is not None
    if not results['forward_pass']:
        print_section("VERIFICATION FAILED")
        print_error("Forward pass failed. Cannot proceed.")
        return False
    
    # Test 4: Loss computation
    results['loss_computation'] = test_loss_computation(outputs, dummy_input, device)
    if not results['loss_computation']:
        print_section("VERIFICATION FAILED")
        print_error("Loss computation failed.")
        return False
    
    # Test 5: Backward pass
    results['backward_pass'] = test_backward_pass(outputs, dummy_input, device)
    if not results['backward_pass']:
        print_section("VERIFICATION FAILED")
        print_error("Backward pass failed.")
        return False
    
    # Test 6: Device handling
    results['device_handling'] = test_device_handling(model, device)
    if not results['device_handling']:
        print_section("VERIFICATION FAILED")
        print_error("Device handling failed.")
        return False
    
    # Final report
    print_section("VERIFICATION SUMMARY")
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} | {test_name}")
    
    print()
    if all_passed:
        print_success("ALL TESTS PASSED - Pipeline is ready for training!")
        return True
    else:
        print_error("SOME TESTS FAILED - Review errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
