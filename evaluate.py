import os
import cv2
import time
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from config import Config
from train import train_zero_shot

def rgb_to_hsv_tensor(rgb_tensor):
    r, g, b = rgb_tensor[:, 0:1, :, :], rgb_tensor[:, 1:2, :, :], rgb_tensor[:, 2:3, :, :]
    max_c, _ = torch.max(rgb_tensor, dim=1, keepdim=True)
    min_c, _ = torch.min(rgb_tensor, dim=1, keepdim=True)
    v = max_c
    return v

def evaluate_zesrd():
    config = Config
    config.init_dirs()
    
    low_dir = 'LOLdataset/eval15/low'
    high_dir = 'LOLdataset/eval15/high'
    out_dir = os.path.join(config.RESULTS_DIR, 'enhanced_images')
    os.makedirs(out_dir, exist_ok=True)
    
    low_files = sorted([f for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    high_files = sorted([f for f in os.listdir(high_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    psnrs, ssims = [], []
    times = []
    
    print("\n" + "="*70)
    print("🚀 ZESRD-HSV Evaluation on LOL eval15")
    print("="*70)
    print(f"Images: {len(low_files)} | Device: {config.DEVICE}")
    print("-"*70)
    
    for idx, fl in enumerate(low_files):
        low_path = os.path.join(low_dir, fl)
        high_path = os.path.join(high_dir, high_files[idx])
        
        img_bgr = cv2.imread(low_path)
        high_bgr = cv2.imread(high_path)
        H, W, _ = img_bgr.shape
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_tensor = transforms.ToTensor()(Image.fromarray(img_rgb)).unsqueeze(0).to(config.DEVICE)
        
        v_tensor = rgb_to_hsv_tensor(img_rgb_tensor)
        
        print(f"\nImage {idx+1:02d}: {fl}")
        
        start_time = time.time()
        v_enhanced, loss_history = train_zero_shot(v_tensor, config)
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        v_enh_np = v_enhanced.squeeze().cpu().numpy()
        v_enh_resized = cv2.resize(v_enh_np, (W, H), interpolation=cv2.INTER_CUBIC)
        v_enh_scaled = (v_enh_resized * 255.0).astype(np.float32)
        
        hsv_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv_orig)
        s_adj = np.clip(s * 1.1, 0, 255).astype(np.float32)
        
        hsv_enh = cv2.merge([h.astype(np.uint8), s_adj.astype(np.uint8), v_enh_scaled.astype(np.uint8)])
        bgr_enh = cv2.cvtColor(hsv_enh, cv2.COLOR_HSV2BGR)
        
        cv2.imwrite(os.path.join(out_dir, fl), bgr_enh)
        
        psnr = psnr_metric(high_bgr, bgr_enh, data_range=255)
        ssim = ssim_metric(high_bgr, bgr_enh, channel_axis=2, data_range=255)
        psnrs.append(psnr)
        ssims.append(ssim)
        
        print(f"  ✅ PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | Time: {elapsed:.2f}s")
    
    print("\n" + "-"*70)
    print(f"📊 FINAL RESULTS:")
    print(f"   MEAN PSNR: {np.mean(psnrs):.2f} dB")
    print(f"   MEAN SSIM: {np.mean(ssims):.4f}")
    print(f"   MEAN TIME: {np.mean(times):.2f} seconds")
    print("-"*70)
    
    return psnrs, ssims

if __name__ == "__main__":
    if os.path.exists('LOLdataset/eval15/low'):
        evaluate_zesrd()
    else:
        print("❌ Dataset not found!")
