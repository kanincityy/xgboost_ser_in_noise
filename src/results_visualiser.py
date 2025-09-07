import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import librosa

from config import (
    PLOTS_PATH, METRICS_PATH, OUTPUT_PATH, DATA_PATH, SNR_LEVELS,
    TARGET_SR, N_FFT, HOP_LENGTH
)
from utils import add_noise, plot_audio_ser

def plot_per_emotion_degradation(results, le):
    """Generates subplots for per-emotion F1-score degradation with CIs."""
    plot_data = []
    conditions = ['Clean'] + sorted(SNR_LEVELS, reverse=True)
    
    for cond in conditions:
        if cond in results:
            for emotion in le.classes_:
                metrics = results[cond]['ci_metrics'][emotion]
                plot_data.append({
                    'Condition': cond,
                    'Emotion': emotion,
                    'Mean_F1': metrics['mean_f1'],
                    'Lower_F1': metrics['lower_f1'],
                    'Upper_F1': metrics['upper_f1'],
                })
    
    df = pd.DataFrame(plot_data)
    
    n_emotions = len(le.classes_)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle('F1-Score Degradation per Emotion with 95% CI', fontsize=20)

    for i, emotion in enumerate(le.classes_):
        ax = axes[i]
        emotion_df = df[df['Emotion'] == emotion]
        
        clean_data = emotion_df[emotion_df['Condition'] == 'Clean'].iloc[0]
        noisy_data = emotion_df[emotion_df['Condition'] != 'Clean']
        noisy_data['Condition'] = pd.to_numeric(noisy_data['Condition'])
        noisy_data = noisy_data.sort_values(by='Condition', ascending=False)
        
        # Plot Clean Baseline
        ax.axhline(clean_data['Mean_F1'], color='blue', linestyle='--', label='Clean Baseline Mean')
        ax.axhspan(clean_data['Lower_F1'], clean_data['Upper_F1'], color='lightblue', alpha=0.3, label='Clean Baseline 95% CI')
        
        # Plot Degradation
        ax.plot(noisy_data['Condition'], noisy_data['Mean_F1'], marker='o', color='black', label='Degradation Mean')
        ax.fill_between(noisy_data['Condition'], noisy_data['Lower_F1'], noisy_data['Upper_F1'], color='gray', alpha=0.5, label='Degradation 95% CI')
        
        ax.set_title(emotion.capitalize(), fontsize=16)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.invert_xaxis()
        if i >= 2: ax.set_xlabel('Condition (SNR dB)', fontsize=12)
        if i % 2 == 0: ax.set_ylabel('F1-Score', fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 0.95))
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(os.path.join(PLOTS_PATH, 'per_emotion_f1_degradation.png'))
    plt.close()
    print("Per-emotion F1-score degradation plot saved.")

def plot_normalised_confusion_matrices(results, le):
    """Plots row-normalised confusion matrices."""
    conditions_to_plot = ['Clean', 20, 5, -5]
    for cond in conditions_to_plot:
        if cond in results:
            cm = results[cond]['cm']
            cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_normalised, annot=True, fmt='.2f', cmap='viridis', 
                        xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title(f'Normalised Confusion Matrix - {cond} Condition')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(os.path.join(PLOTS_PATH, f'cm_normalised_{str(cond).replace("-", "neg")}.png'))
            plt.close()
    print('Normalised confusion matrices saved.')

def visualise_representative_speaker(speaker_id):
    """Generates and saves audio visualisations for the representative speaker."""
    if speaker_id is None:
        print("No representative speaker ID found. Skipping visualisation.")
        return
        
    print(f"\n--- Generating visualisations for representative speaker: {speaker_id} ---")
    test_df = pd.read_pickle(os.path.join(DATA_PATH, 'test_df.pkl'))
    speaker_files = test_df[test_df['speaker_id'] == speaker_id]
    
    conditions = {'Clean': None, 20: 20, 5: 5, -5: -5}

    for _, row in speaker_files.iterrows():
        emotion = row['emotion_label']
        clean_signal, _ = librosa.load(row['full_path'], sr=TARGET_SR, mono=True)
        
        for name, snr in conditions.items():
            signal_to_plot = add_noise(clean_signal, snr) if snr is not None else clean_signal
            
            title = f"Emotion: {emotion.capitalize()} | Speaker: {speaker_id} | Condition: {name}"
            save_name = f"speaker_{speaker_id}_{emotion}_{str(name).replace('-', 'neg')}.png"
            save_path = os.path.join(PLOTS_PATH, 'representative_speaker', save_name)
            
            plot_audio_ser(
                audio_signal=signal_to_plot,
                sample_rate=TARGET_SR,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=128, n_mfcc=13,
                plot_title=title,
                save_path=save_path
            )

def main():
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    # Load results and auxiliary files
    results = joblib.load(os.path.join(METRICS_PATH, 'evaluation_results.pkl'))
    le = joblib.load(os.path.join(DATA_PATH, "emotion_label_encoder.joblib"))
    with open(os.path.join(METRICS_PATH, 'representative_speaker.json'), 'r') as f:
        rep_speaker_id = json.load(f)['speaker_id']

    # Generate and save all plots
    plot_per_emotion_degradation(results, le)
    plot_normalised_confusion_matrices(results, le)
    visualise_representative_speaker(rep_speaker_id)

if __name__ == '__main__':
    main()