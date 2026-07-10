import torch
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler

from models.zesrd_hsv import ZESRD_HSV
from losses.total_loss import TotalLoss
from utils.ema import EMA


def train_zero_shot(v_tensor, config=None):

    if config is None:
        config = Config


    model = ZESRD_HSV(
        base_channels=config.BASE_CHANNELS,
        K=config.K_CURVE
    ).to(config.DEVICE)



    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9,0.999),
        weight_decay=1e-4
    )


    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=300,
        eta_min=1e-6
    )


    criterion = TotalLoss(config)


    scaler = GradScaler()


    ema = EMA(
        model,
        decay=0.999
    )


    best_loss=float("inf")

    best_state=None

    patience=40

    counter=0



    history={

        "loss":[],
        "gamma":[]

    }



    model.train()



    for i in range(300):


        optimizer.zero_grad(
            set_to_none=True
        )


        with autocast():


            outputs=model(v_tensor)


            loss,loss_dict=criterion(
                v_tensor,
                outputs
            )



        scaler.scale(loss).backward()



        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )



        scaler.step(
            optimizer
        )


        scaler.update()



        scheduler.step()



        ema.update()



        history["loss"].append(
            loss.item()
        )



        if "gamma" in outputs:

            history["gamma"].append(
                outputs["gamma"].mean().item()
            )



        if loss.item()<best_loss:


            best_loss=loss.item()


            best_state={
                k:v.clone()
                for k,v in model.state_dict().items()
            }


            counter=0


        else:

            counter+=1



        if counter>=patience:

            print(
                f"Early stopping {i}"
            )

            break



        if (i+1)%50==0:

            print(
                f"Iter {i+1}/300 Loss {loss.item():.5f}"
            )



    model.load_state_dict(
        best_state
    )


    model.eval()


    with torch.no_grad():


        outputs=model(
            v_tensor
        )


        enhanced=torch.clamp(
            outputs["J_scat"],
            0,
            1
        )



    return {

        "image":enhanced,

        "history":history,

        "best_loss":best_loss,

        "model":model

    }
