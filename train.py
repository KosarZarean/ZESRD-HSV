import torch
import torch.optim as optim
from models.zesrd_hsv import ZESRD_HSV
from losses.total_loss import TotalLoss
from config import Config

def train_zero_shot(v_tensor, config=None):
    """Zero-shot training on a single image"""
    if config is None:
        config = Config
    
    model = ZESRD_HSV(
        base_channels=config.BASE_CHANNELS,
        K=config.K_CURVE
    ).to(config.DEVICE)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.TOTAL_ITERATIONS,
        eta_min=1e-6
    )
    
    criterion = TotalLoss(config)
    
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    loss_history = {'total': [], 'rec': [], 'R_cons': [], 'v_cons': [], 'ist': [], 'prior': [], 'vgg': []}
    
    model.train()
    
    for i in range(config.TOTAL_ITERATIONS):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(v_tensor)
        loss, loss_dict = criterion(v_tensor, outputs)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        for key in loss_history.keys():
            if key in loss_dict:
                loss_history[key].append(loss_dict[key])
            else:
                loss_history[key].append(0.0)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter > config.PATIENCE:
            print(f"  Early stopping at iteration {i+1}")
            break
        
        if (i + 1) % 50 == 0:
            print(f"  Iter {i+1:03d}/{config.TOTAL_ITERATIONS}: Loss={loss.item():.4f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        outputs = model(v_tensor)
        v_enhanced = torch.clamp(outputs['J_scat'], 0, 1)
    
    return v_enhanced, loss_history
