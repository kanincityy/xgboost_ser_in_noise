import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from config import (
    PLOTS_PATH, METRICS_PATH, OUTPUT_PATH, DATA_PATH,
    GLOBAL_FEATURES_CLEAN_DIR, GLOBAL_FEATURES_NOISY_DIR,
    SNR_LEVELS, N_BOOTSTRAPS, CONFIDENCE_LEVEL, SPEAKER_ID_TO_VISUALIZE
)
from utils import plot_audio_ser, add_noise # Reusing utility functions

# --- PLOTTING FUNCTIONS --- 

def plot_performance_degradation(results, le, metric='accuracy'):
    """Plots overall model performance degradation vs. SNR with CIs."""
    
    conditions = ['Clean'] + sorted(SNR_LEVELS, reverse=True)
    mean_scores = []
    
    for cond in conditions:
        if cond in results:
            if metric == 'accuracy':
                mean_scores.append(results[cond]['accuracy'])
            elif metric == 'f1-score':
                mean_scores.append(results[cond]['report']['macro avg']['f1-score'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(conditions)), mean_scores, marker='o', linestyle='-')
    plt.axhline(y=1/len(le.classes_), color='r', linestyle='--', label=f'Chance Level ({1/len(le.classes_):.2f})')
    
    plt.xticks(range(len(conditions)), labels=conditions)
    plt.xlabel('Condition (SNR dB)')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Overall Model {metric.title()} vs. Condition')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_PATH, f'{metric}_degradation.png'))
    plt.close()
    print(f'{metric.title()} degradation plot saved.')

def plot_confusion_matrices(results, le):
    """Plots confusion matrices for clean and selected noisy conditions."""
    conditions_to_plot = ['Clean', 20, 5, -5]
    for cond in conditions_to_plot:
        if cond in results:
            cm = results[cond]['cm']
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                        xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title(f'Confusion Matrix - {cond} Condition')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(os.path.join(PLOTS_PATH, f'cm_{str(cond).replace("-", "neg")}.png'))
            plt.close()
    print('Confusion matrices saved.')

def plot_feature_importance(model, feature_names):
    """Plots the top 15 most important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    
    plt.figure(figsize=(10, 8))
    plt.title('Top 15 Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'feature_importance.png'))
    plt.close()
    print('Feature importance plot saved.')

def main():
    """Main function to generate and save all plots."""
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    # 1. Load results, model, and feature names
    results = joblib.load(os.path.join(METRICS_PATH, 'evaluation_results.pkl'))
    model = joblib.load(os.path.join(OUTPUT_PATH, 'best_xgb_model.joblib'))
    feature_names = joblib.load(os.path.join(DATA_PATH, 'feature_column_names.joblib'))
    le = joblib.load(os.path.join(DATA_PATH, "emotion_label_encoder.joblib"))

    # 2. Generate plots
    plot_performance_degradation(results, le, metric='accuracy')
    plot_performance_degradation(results, le, metric='f1-score')
    plot_confusion_matrices(results, le)
    plot_feature_importance(model, feature_names)

if __name__ == '__main__':
    main()