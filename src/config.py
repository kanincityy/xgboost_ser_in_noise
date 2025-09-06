import os

# --- PATHS ---
# Base directory for the project
DRIVE_PATH = './' # Assuming you run from the root of the project
DATA_PATH = os.path.join(DRIVE_PATH, 'data')
OUTPUT_PATH = os.path.join(DRIVE_PATH, 'models')
RESULTS_PATH = os.path.join(DRIVE_PATH, 'results')
METRICS_PATH = os.path.join(RESULTS_PATH, 'metrics')
PLOTS_PATH = os.path.join(RESULTS_PATH, 'plots')

# Subfolders for specific data types
GLOBAL_FEATURES_CLEAN_DIR = os.path.join(DATA_PATH, 'xgb_global_features_scaled_clean')
GLOBAL_FEATURES_NOISY_DIR = os.path.join(DATA_PATH, 'xgb_global_features_scaled_noisy')

# --- DATASET & PREPROCESSING ---
EMOTIONS_TO_KEEP = {
    "02": "calm", 
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

# --- FEATURE EXTRACTION ---
TARGET_SR = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MFCC = 13

# --- NOISE GENERATION ---
SNR_LEVELS = [20, 15, 10, 5, 0, -5, -10]

# --- MODEL TRAINING ---
# Data split ratios
TEST_SIZE_VAL_TEST = 0.3 # 30% for validation + test
TEST_SIZE_FINAL = 0.5   # 50% of the 30% goes to the final test set (i.e., 15% of total)

# BayesSearchCV parameters
N_ITER_BAYES = 50
CV_SPLITS_BAYES = 3

# XGBoost fixed parameters
XGB_FIXED_PARAMS = {
    'objective': 'multi:softmax',
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'random_state': 42
}

# --- VISUALISATION ---
# Qualitative analysis speaker ID (set to None to auto-detect)
SPEAKER_ID_TO_VISUALIZE = None # Will be identified in evaluation script

# Bootstrapping for Confidence Intervals
N_BOOTSTRAPS = 1000
CONFIDENCE_LEVEL = 95