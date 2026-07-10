import torch
from models.zesrd_hsv import ZESRD_HSV
from losses.total_loss import TotalLoss

# ۱. ساخت مدل
model = ZESRD_HSV().cuda()
# ۲. ساخت ورودی مجازی (بسیار مهم: چک کردن ابعاد)
dummy_img = torch.randn(1, 3, 256, 256).cuda()

try:
    # تست خروجی مدل
    outputs = model(dummy_img)
    print("✅ خروجی مدل سالم است. کلیدها:", outputs.keys())

    # تست محاسبه Loss
    criterion = TotalLoss()
    loss = criterion(dummy_img, outputs)
    print(f"✅ محاسبه Loss با موفقیت انجام شد. مقدار: {loss.item()}")

except Exception as e:
    print(f"❌ خطا رخ داد: {e}")
