import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import joblib
from config import (
    DATA_PATH, TARGET_SR, N_MFCC, SNR_LEVELS, GLOBAL_FEATURES_NOISY_DIR
)
from utils import add_noise
from feature_extraction import extract_global_features # Reuse the feature extractor

def main():
    """Generates noisy test sets for all SNR levels."""
    os.makedirs(GLOBAL_FEATURES_NOISY_DIR, exist_ok=True)

    # Load necessary files
    test_df = pd.read_pickle(os.path.join(DATA_PATH, 'test_df.pkl'))
    scaler = joblib.load(os.path.join(DATA_PATH, "global_features_scaler.joblib"))
    le = joblib.load(os.path.join(DATA_PATH, "emotion_label_encoder.joblib"))
    train_feature_means = pd.read_pickle(os.path.join(DATA_PATH, "train_global_feature_means.pkl"))
    feature_names = joblib.load(os.path.join(DATA_PATH, 'feature_column_names.joblib'))

    for snr_db in SNR_LEVELS:
        print(f"\n--- Processing for SNR = {snr_db} dB ---")
        all_noisy_features = []
        labels_list = []
        
        for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc=f"SNR {snr_db} dB"):
            try:
                clean_signal, sr = librosa.load(row['full_path'], sr=TARGET_SR, mono=True)
                noisy_signal = add_noise(clean_signal, snr_db)
                features = extract_global_features(noisy_signal, sr, N_MFCC)
                all_noisy_features.append(features)
                labels_list.append(row['emotion_label'])
            except Exception as e:
                print(f"Error processing {row['full_path']} for SNR {snr_db}: {e}")
        
        if not all_noisy_features:
            print(f"No features extracted for SNR {snr_db}. Skipping.")
            continue

        X_noisy_df = pd.DataFrame(all_noisy_features)[feature_names] # Ensure column order
        X_noisy_df.fillna(train_feature_means, inplace=True)
        
        X_noisy_scaled = scaler.transform(X_noisy_df)
        y_noisy_encoded = le.transform(labels_list)

        # Save the noisy data
        np.save(os.path.join(GLOBAL_FEATURES_NOISY_DIR, f'X_test_snr_{snr_db}.npy'), X_noisy_scaled)
        np.save(os.path.join(GLOBAL_FEATURES_NOISY_DIR, f'y_test_snr_{snr_db}.npy'), y_noisy_encoded)
        print(f"Saved noisy test set for SNR {snr_db} dB.")

if __name__ == '__main__':
    main()