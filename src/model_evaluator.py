import os
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils import resample
from tqdm import tqdm

from config import (
    OUTPUT_PATH, DATA_PATH, GLOBAL_FEATURES_CLEAN_DIR, GLOBAL_FEATURES_NOISY_DIR,
    METRICS_PATH, SNR_LEVELS, N_BOOTSTRAPS, CONFIDENCE_LEVEL
)

def get_bootstrap_metrics(y_true, y_pred, le):
    """Calculates bootstrap confidence intervals for per-emotion metrics."""
    n_classes = len(le.classes_)
    metrics_ci = {}

    for emotion_idx, emotion_name in enumerate(le.classes_):
        per_emotion_f1_scores = []
        per_emotion_acc_scores = []
        
        for _ in range(N_BOOTSTRAPS):
            indices = resample(np.arange(len(y_true)), replace=True)
            boot_true, boot_pred = y_true[indices], y_pred[indices]
            
            # F1-Score
            f1 = f1_score(boot_true, boot_pred, labels=[emotion_idx], average='macro', zero_division=0)
            per_emotion_f1_scores.append(f1)
            
            # Accuracy (One-vs-Rest)
            true_binary = (boot_true == emotion_idx).astype(int)
            pred_binary = (boot_pred == emotion_idx).astype(int)
            acc = accuracy_score(true_binary, pred_binary)
            per_emotion_acc_scores.append(acc)

        lower_bound = (100 - CONFIDENCE_LEVEL) / 2
        upper_bound = 100 - lower_bound
        
        metrics_ci[emotion_name] = {
            'mean_f1': np.mean(per_emotion_f1_scores),
            'lower_f1': np.percentile(per_emotion_f1_scores, lower_bound),
            'upper_f1': np.percentile(per_emotion_f1_scores, upper_bound),
            'mean_accuracy': np.mean(per_emotion_acc_scores),
            'lower_accuracy': np.percentile(per_emotion_acc_scores, lower_bound),
            'upper_accuracy': np.percentile(per_emotion_acc_scores, upper_bound),
        }
    return metrics_ci

def find_representative_speaker(model, test_df, le):
    """Finds speaker whose performance trend best matches the overall average."""
    test_speaker_ids = sorted(test_df['speaker_id'].unique())
    speaker_id_array = test_df['speaker_id'].values
    per_speaker_accuracies = {sp_id: [] for sp_id in test_speaker_ids}
    overall_accuracies = []

    for snr in tqdm(SNR_LEVELS, desc="Finding Rep. Speaker"):
        X_noisy = np.load(os.path.join(GLOBAL_FEATURES_NOISY_DIR, f'X_test_snr_{snr}.npy'))
        y_true_noisy = np.load(os.path.join(GLOBAL_FEATURES_NOISY_DIR, f'y_test_snr_{snr}.npy'))
        y_pred_noisy = model.predict(X_noisy)
        
        overall_accuracies.append(accuracy_score(y_true_noisy, y_pred_noisy))

        for speaker_id in test_speaker_ids:
            mask = (speaker_id_array == speaker_id)
            if np.sum(mask) > 0:
                speaker_acc = accuracy_score(y_true_noisy[mask], y_pred_noisy[mask])
                per_speaker_accuracies[speaker_id].append(speaker_acc)
            else:
                per_speaker_accuracies[speaker_id].append(np.nan)

    speaker_errors = {}
    overall_trend = np.array(overall_accuracies)
    for speaker_id, trend in per_speaker_accuracies.items():
        speaker_trend = np.array(trend)
        if not np.isnan(speaker_trend).any():
            mse = np.mean((speaker_trend - overall_trend) ** 2)
            speaker_errors[speaker_id] = mse
            
    best_speaker_id = min(speaker_errors, key=speaker_errors.get) if speaker_errors else None
    print(f"\nRepresentative speaker analysis complete. Best speaker ID: {best_speaker_id}")
    return best_speaker_id

def main():
    os.makedirs(METRICS_PATH, exist_ok=True)
    
    # Load model, data splits, and encoder
    model = joblib.load(os.path.join(OUTPUT_PATH, 'best_xgb_model.joblib'))
    test_df = pd.read_pickle(os.path.join(DATA_PATH, 'test_df.pkl'))
    le = joblib.load(os.path.join(DATA_PATH, "emotion_label_encoder.joblib"))
    
    all_results = {}

    # --- Process Clean Data ---
    print("Evaluating on Clean Test Set...")
    X_test_clean = np.load(os.path.join(GLOBAL_FEATURES_CLEAN_DIR, 'X_test_clean.npy'))
    y_test_clean = np.load(os.path.join(GLOBAL_FEATURES_CLEAN_DIR, 'y_test_clean.npy'))
    y_pred_clean = model.predict(X_test_clean)
    
    all_results['Clean'] = {
        'accuracy': accuracy_score(y_test_clean, y_pred_clean),
        'report': classification_report(y_test_clean, y_pred_clean, target_names=le.classes_, output_dict=True, zero_division=0),
        'cm': confusion_matrix(y_test_clean, y_pred_clean),
        'ci_metrics': get_bootstrap_metrics(y_test_clean, y_pred_clean, le)
    }

    # --- Process Noisy Data ---
    for snr_db in SNR_LEVELS:
        print(f"Evaluating on Noisy Test Set (SNR {snr_db} dB)...")
        try:
            X_noisy = np.load(os.path.join(GLOBAL_FEATURES_NOISY_DIR, f'X_test_snr_{snr_db}.npy'))
            y_noisy = np.load(os.path.join(GLOBAL_FEATURES_NOISY_DIR, f'y_test_snr_{snr_db}.npy'))
            y_pred_noisy = model.predict(X_noisy)
            
            all_results[snr_db] = {
                'accuracy': accuracy_score(y_noisy, y_pred_noisy),
                'report': classification_report(y_noisy, y_pred_noisy, target_names=le.classes_, output_dict=True, zero_division=0),
                'cm': confusion_matrix(y_noisy, y_pred_noisy),
                'ci_metrics': get_bootstrap_metrics(y_noisy, y_pred_noisy, le)
            }
        except FileNotFoundError:
            print(f"Data for SNR {snr_db} not found. Skipping.")

    # Find and save representative speaker ID
    best_speaker_id = find_representative_speaker(model, test_df, le)
    with open(os.path.join(METRICS_PATH, 'representative_speaker.json'), 'w') as f:
        json.dump({'speaker_id': best_speaker_id}, f)

    # Save all collected results
    joblib.dump(all_results, os.path.join(METRICS_PATH, 'evaluation_results.pkl'))
    print(f"\nAll evaluation metrics (including CIs) saved.")

if __name__ == '__main__':
    main()