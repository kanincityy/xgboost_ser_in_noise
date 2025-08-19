DISSO_NVMH6.ipynb - Colab.pdf
3,335 lines

pdf

can you refactor this code for me to make it github ready?

Edit
I'll help you refactor this Jupyter notebook code into a well-structured, GitHub-ready Python project. This appears to be a comprehensive Speech Emotion Recognition (SER) research project studying the impact of white noise on emotional cue detection.


Project Structure and Main Files
Code 

Configuration File
Code 

Dataset Handler Module
Code 

Feature Extraction Module
Code 

Noise Generation Utilities
Text 



Continue

Edit

Noise Generation Utilities
Code ∙ Version 2 

XGBoost Model Implementation
Code 

Evaluation and Metrics Module
Code 



Continue

Edit

Evaluation and Metrics Module
Code ∙ Version 2 

Visualization Module
Code 



Continue

Edit

Visualization Module
Code ∙ Version 2 

Main Execution Scripts
Code 




You are out of free messages until 8:00 PM
Continue

"""
Comprehensive visualization module for speech emotion recognition results.

This module provides publication-ready plots for performance analysis,
degradation patterns, feature importance, and qualitative analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)


class SERVisualizer:
    """Comprehensive visualizer for SER experimental results."""
    
    def __init__(self, config: Dict):
        """
        Initialize SER visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.viz_config = config['visualization']
        self.figure_size = tuple(self.viz_config['figure_size'])
        self.dpi = self.viz_config['dpi']
        self.format = self.viz_config['format']
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory
        self.output_dir = Path(config['paths']['figures_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized visualizer, output dir: {self.output_dir}")
    
    def plot_performance_degradation(
        self,
        clean_performance: Dict,
        noisy_performance: Dict[float, Dict],
        snr_levels: List[float],
        emotions: List[str],
        save_path: str = None
    ) -> None:
        """
        Plot overall performance degradation across SNR levels.
        
        Args:
            clean_performance: Clean data performance metrics
            noisy_performance: Noisy data performance per SNR
            snr_levels: List of SNR levels
            emotions: List of emotion labels
            save_path: Optional save path
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        snr_labels = ['Clean'] + [f'{snr}' for snr in sorted(snr_levels, reverse=True)]
        
        # Accuracy plot
        acc_means = [clean_performance['accuracy']['mean']]
        acc_lowers = [clean_performance['accuracy']['lower_ci']]
        acc_uppers = [clean_performance['accuracy']['upper_ci']]
        
        for snr in sorted(snr_levels, reverse=True):
            if snr in noisy_performance:
                acc_means.append(noisy_performance[snr]['accuracy']['mean'])
                acc_lowers.append(noisy_performance[snr]['accuracy']['lower_ci'])
                acc_uppers.append(noisy_performance[snr]['accuracy']['upper_ci'])
        
        x_pos = range(len(snr_labels))
        
        # Plot accuracy
        ax1.plot(x_pos[1:], acc_means[1:], 'o-', linewidth=2, markersize=8, label='Degradation')
        ax1.fill_between(x_pos[1:], acc_lowers[1:], acc_uppers[1:], alpha=0.3)
        
        # Clean baseline
        ax1.axhline(y=acc_means[0], color='blue', linestyle='--', linewidth=2, label='Clean Baseline')
        ax1.fill_between(x_pos, [acc_lowers[0]] * len(x_pos), [acc_uppers[0]] * len(x_pos), 
                        color='skyblue', alpha=0.2)
        
        # Chance level
        chance_level = 1.0 / len(emotions)
        ax1.axhline(y=chance_level, color='red', linestyle=':', linewidth=2, label='Chance Level')
        
        ax1.set_title('Overall Accuracy Degradation', fontsize=14)
        ax1.set_xlabel('Condition (SNR dB)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(snr_labels)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # F1-Score plot (similar structure)
        f1_means = [clean_performance['f1_macro']['mean']]
        f1_lowers = [clean_performance['f1_macro']['lower_ci']]
        f1_uppers = [clean_performance['f1_macro']['upper_ci']]
        
        for snr in sorted(snr_levels, reverse=True):
            if snr in noisy_performance:
                f1_means.append(noisy_performance[snr]['f1_macro']['mean'])
                f1_lowers.append(noisy_performance[snr]['f1_macro']['lower_ci'])
                f1_uppers.append(noisy_performance[snr]['f1_macro']['upper_ci'])
        
        ax2.plot(x_pos[1:], f1_means[1:], 'o-', linewidth=2, markersize=8, label='Degradation')
        ax2.fill_between(x_pos[1:], f1_lowers[1:], f1_uppers[1:], alpha=0.3)
        ax2.axhline(y=f1_means[0], color='blue', linestyle='--', linewidth=2, label='Clean Baseline')
        ax2.fill_between(x_pos, [f1_lowers[0]] * len(x_pos), [f1_uppers[0]] * len(x_pos), 
                        color='skyblue', alpha=0.2)
        
        ax2.set_title('Overall F1-Score Degradation', fontsize=14)
        ax2.set_xlabel('Condition (SNR dB)', fontsize=12)
        ax2.set_ylabel('F1-Score (Macro)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(snr_labels)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Performance degradation plot saved to {save_path}")
        else:
            plt.savefig(self.output_dir / f'performance_degradation.{self.format}', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_per_emotion_performance(
        self,
        per_emotion_results: Dict[str, Dict],
        snr_levels: List[float],
        save_path: str = None
    ) -> None:
        """
        Plot per-emotion performance across conditions.
        
        Args:
            per_emotion_results: Per-emotion results across conditions
            snr_levels: List of SNR levels
            save_path: Optional save path
        """
        emotions = list(next(iter(per_emotion_results.values())).keys())
        n_emotions = len(emotions)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
        axes = axes.flatten()
        
        conditions = ['Clean'] + [str(snr) for snr in sorted(snr_levels, reverse=True)]
        x_pos = range(len(conditions))
        
        for i, emotion in enumerate(emotions):
            ax = axes[i]
            
            # Extract data for this emotion
            acc_means, acc_lowers, acc_uppers = [], [], []
            
            for condition in per_emotion_results:
                if emotion in per_emotion_results[condition]:
                    emotion_data = per_emotion_results[condition][emotion]
                    acc_means.append(emotion_data['accuracy']['mean'])
                    acc_lowers.append(emotion_data['accuracy']['lower_ci'])
                    acc_uppers.append(emotion_data['accuracy']['upper_ci'])
            
            if len(acc_means) == len(conditions):
                # Plot degradation (skip clean for line plot)
                ax.plot(x_pos[1:], acc_means[1:], 'o-', linewidth=2, markersize=6, 
                       color='darkgreen', label='Degradation')
                ax.fill_between(x_pos[1:], acc_lowers[1:], acc_uppers[1:], 
                               alpha=0.3, color='green')
                
                # Clean baseline
                ax.axhline(y=acc_means[0], color='blue', linestyle='--', 
                          linewidth=2, label='Clean Baseline')
                ax.fill_between(x_pos, [acc_lowers[0]] * len(x_pos), [acc_uppers[0]] * len(x_pos), 
                               color='skyblue', alpha=0.2)
            
            # Chance level for binary classification
            ax.axhline(y=0.5, color='red', linestyle=':', linewidth=2, label='Chance Level')
            
            ax.set_title(f'{emotion.capitalize()}', fontsize=14)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(conditions, rotation=45)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend()
        
        # Remove empty subplot if odd number of emotions
        if n_emotions < len(axes):
            fig.delaxes(axes[-1])
        
        fig.suptitle('Per-Emotion Performance Degradation', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f'confusion_matrices.{self.format}', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(
        self,
        feature_importance_df: pd.DataFrame,
        top_n: int = None,
        save_path: str = None
    ) -> None:
        """
        Plot feature importance from trained model.
        
        Args:
            feature_importance_df: DataFrame with feature importance
            top_n: Number of top features to plot
            save_path: Optional save path
        """
        if top_n is None:
            top_n = self.viz_config['top_n_features']
        
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=self.figure_size)
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', ha='left', va='center', fontsize=10)
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score (Weight)', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance (XGBoost)', fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f'feature_importance.{self.format}', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_audio_analysis(
        self,
        audio_signal: np.ndarray,
        sample_rate: int = 16000,
        title: str = "Audio Analysis",
        save_path: str = None
    ) -> None:
        """
        Plot comprehensive audio analysis (waveform, spectrogram, MFCCs).
        
        Args:
            audio_signal: Audio signal
            sample_rate: Sample rate
            title: Plot title
            save_path: Optional save path
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Waveform
        librosa.display.waveshow(audio_signal, sr=sample_rate, ax=axes[0])
        axes[0].set_title('Waveform', fontsize=12)
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Mel Spectrogram
        S = librosa.feature.melspectrogram(y=audio_signal, sr=sample_rate, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        img1 = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', 
                                       sr=sample_rate, ax=axes[1])
        axes[1].set_title('Mel Spectrogram', fontsize=12)
        axes[1].set_ylabel('Frequency (Mel)')
        fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
        img2 = librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate, ax=axes[2])
        axes[2].set_title('MFCCs', fontsize=12)
        axes[2].set_ylabel('MFCC Coefficients')
        axes[2].set_xlabel('Time (s)')
        fig.colorbar(img2, ax=axes[2])
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()  # Close to prevent display in notebook
        else:
            plt.show()
    
    def create_degradation_sequence(
        self,
        clean_audio_path: str,
        snr_levels: List[float],
        emotion: str,
        speaker_id: int,
        noise_generator,
        target_sr: int = 16000
    ) -> None:
        """
        Create visual sequence showing audio degradation.
        
        Args:
            clean_audio_path: Path to clean audio file
            snr_levels: List of SNR levels to demonstrate
            emotion: Emotion label
            speaker_id: Speaker ID
            noise_generator: Noise generator instance
            target_sr: Target sample rate
        """
        import os
        
        # Load clean audio
        clean_signal, sr = librosa.load(clean_audio_path, sr=target_sr, mono=True)
        
        # Create output directory for this sequence
        sequence_dir = self.output_dir / f'degradation_sequence_speaker_{speaker_id}_{emotion}'
        sequence_dir.mkdir(exist_ok=True)
        
        # Plot clean version
        clean_title = f"Emotion: {emotion.capitalize()} | Speaker: {speaker_id} | Condition: Clean"
        clean_save_path = sequence_dir / f'clean.{self.format}'
        self.plot_audio_analysis(clean_signal, target_sr, clean_title, clean_save_path)
        
        # Plot noisy versions
        for snr in snr_levels:
            noisy_signal = noise_generator.add_noise(clean_signal, snr)
            noisy_title = f"Emotion: {emotion.capitalize()} | Speaker: {speaker_id} | Condition: {snr}dB SNR"
            noisy_save_path = sequence_dir / f'snr_{snr}db.{self.format}'
            self.plot_audio_analysis(noisy_signal, target_sr, noisy_title, noisy_save_path)
        
        logger.info(f"Degradation sequence saved to {sequence_dir}")
    
    def plot_robustness_comparison(
        self,
        model_results: Dict[str, Dict],
        metric: str = 'accuracy',
        save_path: str = None
    ) -> None:
        """
        Compare robustness of multiple models.
        
        Args:
            model_results: Results from multiple models
            metric: Metric to compare
            save_path: Optional save path
        """
        plt.figure(figsize=self.figure_size)
        
        for model_name, results in model_results.items():
            if 'degradation_curves' in results and metric in results['degradation_curves']:
                curve = results['degradation_curves'][metric]
                conditions = ['Clean'] + [str(snr) for snr in sorted(results['snr_levels'], reverse=True)]
                
                plt.plot(range(len(curve)), curve, 'o-', linewidth=2, 
                        markersize=6, label=model_name)
        
        plt.xlabel('Condition Index', fontsize=12)
        plt.ylabel(f'{metric.capitalize()}', fontsize=12)
        plt.title(f'Model Robustness Comparison ({metric.capitalize()})', fontsize=14)
        plt.xticks(range(len(conditions)), conditions, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f'robustness_comparison_{metric}.{self.format}', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_statistical_summary(
        self,
        summary_stats: Dict,
        save_path: str = None
    ) -> None:
        """
        Plot statistical summary of experimental results.
        
        Args:
            summary_stats: Statistical summary data
            save_path: Optional save path
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance distribution
        if 'performance_distribution' in summary_stats:
            perf_data = summary_stats['performance_distribution']
            ax1.hist(perf_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(np.mean(perf_data), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(perf_data):.3f}')
            ax1.set_xlabel('Accuracy')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Performance Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Degradation by SNR
        if 'degradation_by_snr' in summary_stats:
            snr_data = summary_stats['degradation_by_snr']
            snrs = list(snr_data.keys())
            degradations = list(snr_data.values())
            
            ax2.bar(range(len(snrs)), degradations, color='coral', alpha=0.7)
            ax2.set_xlabel('SNR Level (dB)')
            ax2.set_ylabel('Performance Drop')
            ax2.set_title('Performance Degradation by SNR')
            ax2.set_xticks(range(len(snrs)))
            ax2.set_xticklabels(snrs)
            ax2.grid(True, alpha=0.3)
        
        # Per-emotion robustness
        if 'emotion_robustness' in summary_stats:
            emotion_data = summary_stats['emotion_robustness']
            emotions = list(emotion_data.keys())
            robustness = list(emotion_data.values())
            
            ax3.barh(range(len(emotions)), robustness, color='lightgreen', alpha=0.7)
            ax3.set_xlabel('Robustness Score')
            ax3.set_ylabel('Emotion')
            ax3.set_title('Per-Emotion Robustness')
            ax3.set_yticks(range(len(emotions)))
            ax3.set_yticklabels(emotions)
            ax3.grid(True, alpha=0.3)
        
        # Feature importance summary
        if 'top_feature_categories' in summary_stats:
            feature_data = summary_stats['top_feature_categories']
            categories = list(feature_data.keys())
            importances = list(feature_data.values())
            
            ax4.pie(importances, labels=categories, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Feature Category Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f'statistical_summary.{self.format}', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()


class PublicationPlotter:
    """Specialized plotter for publication-quality figures."""
    
    def __init__(self, config: Dict):
        """Initialize publication plotter."""
        self.config = config
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'serif',
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.size': 5,
            'ytick.major.size': 5,
            'legend.frameon': False,
            'figure.dpi': 300
        })
    
    def create_main_results_figure(
        self,
        results_data: Dict,
        save_path: str
    ) -> None:
        """
        Create main results figure for publication.
        
        Args:
            results_data: Experimental results
            save_path: Save path for figure
        """
        fig = plt.figure(figsize=(12, 8))
        
        # Create grid layout
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2, 2, 1])
        
        # Main degradation plot
        ax_main = fig.add_subplot(gs[:, :2])
        # Feature importance
        ax_features = fig.add_subplot(gs[0, 2])
        # Confusion matrix
        ax_cm = fig.add_subplot(gs[1, 2])
        
        # Plot main results
        # ... (implement specific publication plots)
        
        # Add panel labels
        ax_main.text(-0.1, 1.05, 'A', transform=ax_main.transAxes, 
                    fontsize=16, fontweight='bold')
        ax_features.text(-0.15, 1.05, 'B', transform=ax_features.transAxes, 
                        fontsize=16, fontweight='bold')
        ax_cm.text(-0.15, 1.05, 'C', transform=ax_cm.transAxes, 
                  fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Publication figure saved to {save_path}")
    
    def create_supplementary_figures(
        self,
        supplementary_data: Dict,
        output_dir: str
    ) -> None:
        """
        Create supplementary figures for publication.
        
        Args:
            supplementary_data: Supplementary experimental data
            output_dir: Output directory for figures
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create individual supplementary figures
        # ... (implement supplementary plots)
        
        logger.info(f"Supplementary figures saved to {output_dir}")per_emotion_performance.{self.format}', 
                       dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrices(
        self,
        confusion_matrices: Dict,
        emotion_labels: List[str],
        conditions: List[str] = None,
        save_path: str = None
    ) -> None:
        """
        Plot confusion matrices for different conditions.
        
        Args:
            confusion_matrices: Dictionary of confusion matrices
            emotion_labels: List of emotion labels
            conditions: List of condition names
            save_path: Optional save path
        """
        if conditions is None:
            conditions = list(confusion_matrices.keys())
        
        n_conditions = len(conditions)
        cols = min(3, n_conditions)
        rows = int(np.ceil(n_conditions / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_conditions == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, condition in enumerate(conditions):
            ax = axes[i] if n_conditions > 1 else axes[0]
            
            if condition in confusion_matrices:
                cm = confusion_matrices[condition]
                
                # Normalize confusion matrix
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                # Plot heatmap
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='viridis',
                           xticklabels=emotion_labels, yticklabels=emotion_labels,
                           ax=ax, cbar_kws={'shrink': 0.8})
                
                ax.set_title(f'Confusion Matrix - {condition}', fontsize=12)
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f'confusion_matrix_{condition}.png', dpi=self.dpi, bbox_inches='tight')