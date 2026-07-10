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



def evaluate_zesrd():

    config = Config
    config.init_dirs()


    low_dir = 'LOLdataset/eval15/low'
    high_dir = 'LOLdataset/eval15/high'


    out_dir = os.path.join(
        config.RESULTS_DIR,
        'enhanced_images'
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )


    low_files = sorted(
        [
            f for f in os.listdir(low_dir)
            if f.endswith(('.png','.jpg','.jpeg'))
        ]
    )


    high_files = sorted(
        [
            f for f in os.listdir(high_dir)
            if f.endswith(('.png','.jpg','.jpeg'))
        ]
    )


    psnrs=[]
    ssims=[]
    times=[]


    print("\n"+"="*70)
    print("🚀 ZESRD-HSV Evaluation on LOL eval15")
    print("="*70)

    print(
        f"Images: {len(low_files)} | Device: {config.DEVICE}"
    )

    print("-"*70)



    transform = transforms.ToTensor()



    for idx, fl in enumerate(low_files):


        low_path=os.path.join(
            low_dir,
            fl
        )


        high_path=os.path.join(
            high_dir,
            high_files[idx]
        )


        img_bgr=cv2.imread(low_path)

        high_bgr=cv2.imread(high_path)


        H,W,_=img_bgr.shape



        img_rgb=cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2RGB
        )


        rgb_tensor=transform(
            Image.fromarray(img_rgb)
        ).unsqueeze(0).to(
            config.DEVICE
        )


        print(
            f"\nImage {idx+1:02d}: {fl}"
        )



        start=time.time()



        result=train_zero_shot(
            rgb_tensor,
            config
        )


        elapsed=time.time()-start

        times.append(elapsed)



        enhanced=result["image"]



        enhanced_np=(
            enhanced
            .squeeze()
            .permute(1,2,0)
            .cpu()
            .numpy()
        )


        enhanced_np=np.clip(
            enhanced_np*255,
            0,
            255
        ).astype(
            np.uint8
        )


        enhanced_bgr=cv2.cvtColor(
            enhanced_np,
            cv2.COLOR_RGB2BGR
        )


        cv2.imwrite(
            os.path.join(out_dir,fl),
            enhanced_bgr
        )



        psnr=psnr_metric(
            cv2.cvtColor(
                high_bgr,
                cv2.COLOR_BGR2RGB
            ),
            enhanced_np,
            data_range=255
        )


        ssim=ssim_metric(
            cv2.cvtColor(
                high_bgr,
                cv2.COLOR_BGR2RGB
            ),
            enhanced_np,
            channel_axis=2,
            data_range=255
        )


        psnrs.append(psnr)
        ssims.append(ssim)



        print(
            f"✅ PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | Time: {elapsed:.2f}s"
        )



    print("\n"+"-"*70)

    print("📊 FINAL RESULTS")

    print(
        f"MEAN PSNR: {np.mean(psnrs):.2f} dB"
    )

    print(
        f"MEAN SSIM: {np.mean(ssims):.4f}"
    )

    print(
        f"MEAN TIME: {np.mean(times):.2f}s"
    )

    print("-"*70)



    return psnrs,ssims



if __name__=="__main__":

    evaluate_zesrd()
