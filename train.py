import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.zesrd_hsv import ZESRD_HSV
from losses.total_loss import TotalLoss
from utils.ema import EMA # فرض بر وجود کلاس EMA در utils
import config

def train_one_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. بارگذاری مدل و انتقال به GPU
    model = ZESRD_HSV().to(device)
    model.train()
    
    # 2. تنظیمات Loss و Optimizer
    criterion = TotalLoss(config=config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # 3. راه اندازی EMA
    ema = EMA(model, decay=0.999)
    ema.register()

    # فرض: تصویر خوانده شده و به [1, 3, H, W] تبدیل شده است
    image = load_image(image_path).to(device) 

    # 4. حلقه آموزش (Zero-shot)
    for epoch in range(200):
        optimizer.zero_grad()
        
        # قرارداد: ورودی [B,3,H,W] و دریافت خروجی به صورت دیکشنری
        outputs = model(image)
        
        # قرارداد: استفاده از دیکشنری در Loss
        loss = criterion(image, outputs)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # بازگرداندن مدل نهایی (با وزن‌های EMA)
    ema.apply_shadow()
    return model
    
