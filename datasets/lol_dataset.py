"""
LOL Dataset Loader for ZESRD-HSV
Downloads and prepares the LOL dataset from Hugging Face
"""

import os
import zipfile
import shutil
import requests
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class LOLDataset(Dataset):
    """LOL Dataset for training (paired low-light and normal-light images)"""
    
    def __init__(self, low_dir, high_dir, transform=None):
        """
        Args:
            low_dir (str): Path to low-light images
            high_dir (str): Path to normal-light images
            transform: Optional transform to apply
        """
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        
        # Get all image files
        self.low_files = sorted([f for f in os.listdir(low_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.high_files = sorted([f for f in os.listdir(high_dir) 
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Ensure matching files
        self.common_files = list(set(self.low_files) & set(self.high_files))
        self.common_files.sort()
        
        print(f"✅ Loaded {len(self.common_files)} image pairs from LOL dataset")
        
    def __len__(self):
        return len(self.common_files)
    
    def __getitem__(self, idx):
        img_name = self.common_files[idx]
        
        low_path = os.path.join(self.low_dir, img_name)
        high_path = os.path.join(self.high_dir, img_name)
        
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)
        
        return {
            'low': low_img,
            'high': high_img,
            'name': img_name
        }


def download_lol_dataset(target_dir='LOLdataset'):
    """
    Download and extract LOL dataset from Hugging Face
    
    Args:
        target_dir (str): Directory to save the dataset
    """
    
    print("="*60)
    print("📥 Downloading LOL Dataset from Hugging Face")
    print("="*60)
    
    # Clean up old files
    print("[1/5] Cleaning old files...")
    !rm -rf {target_dir} LOLdataset_extracted lol_dataset.zip
    
    # Download
    print("[2/5] Downloading dataset (331 MB)...")
    url = "https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip"
    
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open('lol_dataset.zip', 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))
        print("✅ Download complete!")
        
    except Exception as e:
        print(f"⚠️ Download failed: {e}")
        print("Trying alternative method with wget...")
        !wget -q --show-progress https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip -O lol_dataset.zip
    
    # Extract
    print("[3/5] Extracting dataset...")
    with zipfile.ZipFile('lol_dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('LOLdataset_extracted')
    print("✅ Extraction complete!")
    
    # Restructure
    print("[4/5] Restructuring folders...")
    os.makedirs(f'{target_dir}/our485/low', exist_ok=True)
    os.makedirs(f'{target_dir}/our485/high', exist_ok=True)
    os.makedirs(f'{target_dir}/eval15/low', exist_ok=True)
    os.makedirs(f'{target_dir}/eval15/high', exist_ok=True)
    
    # Find source root
    src_root = None
    for root, dirs, files in os.walk('LOLdataset_extracted'):
        if 'our485' in dirs and 'eval15' in dirs:
            src_root = root
            break
    
    if src_root:
        print(f"✅ Found data at: {src_root}")
        
        # Copy training data
        print("  Copying training data (our485)...")
        for f in os.listdir(os.path.join(src_root, 'our485', 'low')):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(os.path.join(src_root, 'our485', 'low', f), 
                           f'{target_dir}/our485/low/')
        for f in os.listdir(os.path.join(src_root, 'our485', 'high')):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(os.path.join(src_root, 'our485', 'high', f), 
                           f'{target_dir}/our485/high/')
        
        # Copy test data
        print("  Copying test data (eval15)...")
        for f in os.listdir(os.path.join(src_root, 'eval15', 'low')):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(os.path.join(src_root, 'eval15', 'low', f), 
                           f'{target_dir}/eval15/low/')
        for f in os.listdir(os.path.join(src_root, 'eval15', 'high')):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(os.path.join(src_root, 'eval15', 'high', f), 
                           f'{target_dir}/eval15/high/')
    else:
        raise Exception("❌ Dataset structure not found!")
    
    # Clean up
    print("[5/5] Cleaning up temporary files...")
    !rm -rf LOLdataset_extracted lol_dataset.zip
    
    # Verify
    train_count = len(os.listdir(f'{target_dir}/our485/low'))
    test_count = len(os.listdir(f'{target_dir}/eval15/low'))
    
    print("\n" + "-"*40)
    print("🔎 Final Verification:")
    print("-"*40)
    print(f"✅ Training images (our485): {train_count}")
    print(f"✅ Test images (eval15): {test_count}")
    
    if train_count >= 485 and test_count >= 15:
        print("\n🚀 Dataset ready successfully!")
    else:
        print(f"\n⚠️ Warning: Expected 485 train, 15 test images")
    
    return target_dir


def get_lol_dataloader(low_dir, high_dir, batch_size=1, shuffle=False, image_size=256):
    """
    Get DataLoader for LOL dataset
    
    Args:
        low_dir: Path to low-light images
        high_dir: Path to high-light images
        batch_size: Batch size
        shuffle: Shuffle data
        image_size: Resize images to this size
    
    Returns:
        DataLoader object
    """
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    dataset = LOLDataset(low_dir, high_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    return dataloader


# ============================================================
# Quick test function
# ============================================================

def test_lol_dataset():
    """Test if LOL dataset is loaded correctly"""
    
    print("\n" + "="*50)
    print("🧪 Testing LOL Dataset")
    print("="*50)
    
    # Check if dataset exists
    if not os.path.exists('LOLdataset/eval15/low'):
        print("❌ Dataset not found. Downloading...")
        download_lol_dataset()
    
    # Load test images
    low_dir = 'LOLdataset/eval15/low'
    high_dir = 'LOLdataset/eval15/high'
    
    if os.path.exists(low_dir):
        files = os.listdir(low_dir)
        print(f"\n✅ Found {len(files)} test images")
        print(f"  Sample: {files[0] if files else 'None'}")
        
        # Load a sample image
        if files:
            sample_path = os.path.join(low_dir, files[0])
            img = Image.open(sample_path)
            print(f"  Image size: {img.size}")
            print(f"  Image mode: {img.mode}")
    
    print("\n✅ Dataset test complete!")


if __name__ == "__main__":
    test_lol_dataset()
