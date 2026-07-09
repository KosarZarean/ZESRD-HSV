import sys
import os
sys.path.append('.')

from config import Config
from evaluate import evaluate_zesrd

def main():
    print("="*70)
    print("🚀 ZESRD-HSV: Zero-Shot Image Enhancement")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    print(f"Total Iterations: {Config.TOTAL_ITERATIONS}")
    
    if os.path.exists('LOLdataset/eval15/low'):
        evaluate_zesrd()
    else:
        print("\n❌ Dataset not found!")
        print("Please download the LOL dataset first.")
        print("Dataset should be at: LOLdataset/eval15/low")

if __name__ == "__main__":
    main()
