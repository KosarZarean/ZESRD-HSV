import os
import torch

class Config:
    # Hardware
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    
    # Architecture
    BASE_CHANNELS = 32
    K_CURVE = 8
    MSR_SIGMAS = [15, 80, 250]
    
    # Optimization
    TOTAL_ITERATIONS = 300
    LEARNING_RATE = 0.002
    WEIGHT_DECAY = 1e-5
    PATIENCE = 50
    
    # Loss Weights (Table 1)
    LAMBDA_REC = 1.0
    LAMBDA_R_CONS = 0.6
    LAMBDA_V_CONS = 0.3
    LAMBDA_A_CONS = 0.1
    LAMBDA_IST = 0.1
    LAMBDA_PRIOR = 0.05
    LAMBDA_VGG = 0.05
    
    # Dataset paths
    DATASETS = {
        'LOL_eval15': {
            'low': 'LOLdataset/eval15/low',
            'high': 'LOLdataset/eval15/high'
        }
    }
    
    # Output paths
    RESULTS_DIR = 'results'
    CHECKPOINT_DIR = 'checkpoints'
    VISUAL_DIR = os.path.join(RESULTS_DIR, 'visual_comparisons')
    
    @classmethod
    def init_dirs(cls):
        for path in [cls.RESULTS_DIR, cls.CHECKPOINT_DIR, cls.VISUAL_DIR]:
            os.makedirs(path, exist_ok=True)
