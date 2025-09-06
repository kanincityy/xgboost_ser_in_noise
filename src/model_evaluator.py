import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import (
    OUTPUT_PATH, DATA_PATH, GLOBAL_FEATURES_CLEAN_DIR, GLOBAL_FEATURES_NOISY_DIR,
    METRICS_PATH, SNR_LEVELS
)

def evaluate_and_save(model, X, y, le, condition_name):
    """Evaluates the model on a dataset and returns metrics."""
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    print(f"--- Results for {condition_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y, y_pred, target_names=le.classes_, zero_division=0))
    
    return {'accuracy': accuracy, 'report': report, 'cm': cm}

def main():
    """Evaluates the final model on clean and noisy test sets."""
    os.makedirs(METRICS_PATH, exist_ok=True)
    
    # 1. Load model and auxiliary files
    model = joblib.load(os.path.join(OUTPUT_PATH, 'best_xgb_model.joblib'))
    le = joblib.load(os.path.join(DATA_PATH, "emotion_label_encoder.joblib"))
    
    all_results = {}
    
    # 2. Evaluate on clean test data
    X_test_clean = np.load(os.path.join(GLOBAL_FEATURES_CLEAN_DIR, 'X_test_clean.npy'))
    y_test_clean = np.load(os.path.join(GLOBAL_FEATURES_CLEAN_DIR, 'y_test_clean.npy'))
    all_results['Clean'] = evaluate_and_save(model, X_test_clean, y_test_clean, le, 'Clean Test Set')

    # 3. Evaluate on noisy test data
    for snr_db in SNR_LEVELS:
        try:
            X_noisy = np.load(os.path.join(GLOBAL_FEATURES_NOISY_DIR, f'X_test_snr_{snr_db}.npy'))
            y_noisy = np.load(os.path.join(GLOBAL_FEATURES_NOISY_DIR, f'y_test_snr_{snr_db}.npy'))
            all_results[snr_db] = evaluate_and_save(model, X_noisy, y_noisy, le, f'Noisy Test Set (SNR {snr_db} dB)')
        except FileNotFoundError:
            print(f"Noisy data for SNR {snr_db} not found. Skipping.")
    
    # 4. Save results to disk
    results_path = os.path.join(METRICS_PATH, 'evaluation_results.pkl')
    joblib.dump(all_results, results_path)
    print(f"\nAll evaluation metrics saved to {results_path}")

    # Create a summary CSV for quick review
    summary = {
        'Condition': list(all_results.keys()),
        'Accuracy': [res['accuracy'] for res in all_results.values()]
    }
    summary_df = pd.DataFrame(summary).sort_values(by='Condition', key=lambda x: pd.to_numeric(x, errors='coerce'), ascending=False)
    summary_df.to_csv(os.path.join(METRICS_PATH, 'accuracy_summary.csv'), index=False)
    print("Accuracy summary saved to accuracy_summary.csv")

if __name__ == '__main__':
    main()