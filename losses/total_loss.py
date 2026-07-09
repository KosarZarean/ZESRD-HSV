# ============================================================
# Cell: Emergency Run - Without VGG Loss
# ============================================================

import sys
import torch
import gc

# Clear cache
torch.cuda.empty_cache()
gc.collect()

# Force reload
sys.path.append('/content/ZESRD-HSV')
for module in ['losses.perceptual', 'losses.total_loss', 'train', 'evaluate']:
    if module in sys.modules:
        del sys.modules[module]

from config import Config

# ✅ Disable VGG loss
Config.LAMBDA_VGG = 0.0
print("⚠️ VGG Perceptual Loss DISABLED (LAMBDA_VGG = 0.0)")

from evaluate import evaluate_zesrd

print("="*70)
print("🚀 ZESRD-HSV Evaluation (Emergency - No VGG)")
print("="*70)
print(f"Images: 15 | Device: {Config.DEVICE}")
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
