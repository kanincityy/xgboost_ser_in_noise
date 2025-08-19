import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from scipy import stats

logger = logging.getLogger(__name__)


class SERMetricsCalculator:
    """Comprehensive metrics calculator for SER evaluation."""
    
    def __init__(self, config: Dict):
        """
        Initialize metrics calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.eval_config = config['evaluation']
        self.confidence_level = self.eval_config['confidence_level']
        self.n_bootstraps = self.eval_config['n_bootstraps']
        
        logger.info(f"Initialized metrics calculator with {self.confidence_level}% CI")
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0)
        }
        
        return metrics
    
    def bootstrap_confidence_interval(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metric_func: callable,
        **metric_kwargs
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for a metric.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric_func: Metric function to evaluate
            **metric_kwargs: Additional arguments for metric function
            
        Returns:
            Tuple of (mean, lower_ci, upper_ci)
        """
        bootstrap_scores = []
        
        for _ in range(self.n_bootstraps):
            # Bootstrap sample
            indices = resample(np.arange(len(y_true)), n_samples=len(y_true), replace=True)
            boot_true = y_true[indices]
            boot_pred = y_pred[indices]
            
            # Calculate metric
            try:
                score = metric_func(boot_true, boot_pred, **metric_kwargs)
                bootstrap_scores.append(score)
            except Exception as e:
                logger.warning(f"Bootstrap iteration failed: {e}")
                continue
        
        if not bootstrap_scores:
            return 0.0, 0.0, 0.0
        
        # Calculate confidence intervals
        alpha = (100 - self.confidence_level) / 2
        lower_ci = np.percentile(bootstrap_scores, alpha)
        upper_ci = np.percentile(bootstrap_scores, 100 - alpha)
        mean_score = np.mean(bootstrap_scores)
        
        return mean_score, lower_ci, upper_ci
    
    def calculate_per_emotion_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        emotion_labels: List[str]
    ) -> Dict:
        """
        Calculate per-emotion metrics with confidence intervals.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            emotion_labels: List of emotion label names
            
        Returns:
            Dictionary of per-emotion metrics
        """
        per_emotion_metrics = {}
        
        for emotion_idx, emotion_name in enumerate(emotion_labels):
            # Create binary classification problem for this emotion
            true_binary = (y_true == emotion_idx).astype(int)
            pred_binary = (y_pred == emotion_idx).astype(int)
            
            # Calculate metrics with bootstrap CI
            accuracy_mean, accuracy_lower, accuracy_upper = self.bootstrap_confidence_interval(
                true_binary, pred_binary, accuracy_score
            )
            
            f1_mean, f1_lower, f1_upper = self.bootstrap_confidence_interval(
                y_true, y_pred, f1_score, 
                labels=[emotion_idx], average=None, zero_division=0
            )
            
            # Handle case where f1_score returns array
            if isinstance(f1_mean, np.ndarray):
                f1_mean = f1_mean[0] if len(f1_mean) > 0 else 0.0
            if isinstance(f1_lower, np.ndarray):
                f1_lower = f1_lower[0] if len(f1_lower) > 0 else 0.0
            if isinstance(f1_upper, np.ndarray):
                f1_upper = f1_upper[0] if len(f1_upper) > 0 else 0.0
            
            per_emotion_metrics[emotion_name] = {
                'accuracy': {
                    'mean': accuracy_mean,
                    'lower_ci': accuracy_lower,
                    'upper_ci': accuracy_upper
                },
                'f1_score': {
                    'mean': f1_mean,
                    'lower_ci': f1_lower,
                    'upper_ci': f1_upper
                }
            }
        
        return per_emotion_metrics
    
    def calculate_overall_metrics_with_ci(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict:
        """
        Calculate overall metrics with confidence intervals.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of overall metrics with CIs
        """
        overall_metrics = {}
        
        # Accuracy
        acc_mean, acc_lower, acc_upper = self.bootstrap_confidence_interval(
            y_true, y_pred, accuracy_score
        )
        overall_metrics['accuracy'] = {
            'mean': acc_mean,
            'lower_ci': acc_lower,
            'upper_ci': acc_upper
        }
        
        # F1-Score (macro)
        f1_mean, f1_lower, f1_upper = self.bootstrap_confidence_interval(
            y_true, y_pred, f1_score, average='macro', zero_division=0
        )
        overall_metrics['f1_macro'] = {
            'mean': f1_mean,
            'lower_ci': f1_lower,
            'upper_ci': f1_upper
        }
        
        return overall_metrics


class NoiseRobustnessAnalyzer:
    """Analyzer for model robustness to noise degradation."""
    
    def __init__(self, config: Dict):
        """Initialize robustness analyzer."""
        self.config = config
        self.snr_levels = config['noise']['snr_levels']
        self.metrics_calculator = SERMetricsCalculator(config)
    
    def analyze_degradation_patterns(
        self,
        clean_results: Dict,
        noisy_results: Dict[float, Dict]
    ) -> Dict:
        """
        Analyze degradation patterns across SNR levels.
        
        Args:
            clean_results: Results on clean data
            noisy_results: Results on noisy data per SNR level
            
        Returns:
            Degradation analysis results
        """
        logger.info("Analyzing degradation patterns...")
        
        degradation_analysis = {
            'snr_levels': self.snr_levels,
            'clean_baseline': clean_results,
            'noisy_performance': noisy_results,
            'degradation_curves': {},
            'robustness_metrics': {}
        }
        
        # Extract performance curves
        accuracy_curve = [clean_results['accuracy']['mean']]
        f1_curve = [clean_results['f1_macro']['mean']]
        
        for snr in sorted(self.snr_levels, reverse=True):
            if snr in noisy_results:
                accuracy_curve.append(noisy_results[snr]['accuracy']['mean'])
                f1_curve.append(noisy_results[snr]['f1_macro']['mean'])
        
        degradation_analysis['degradation_curves'] = {
            'accuracy': accuracy_curve,
            'f1_macro': f1_curve
        }
        
        # Calculate robustness metrics
        if len(accuracy_curve) > 1:
            # Area under the curve
            auc_accuracy = np.trapz(accuracy_curve) / len(accuracy_curve)
            auc_f1 = np.trapz(f1_curve) / len(f1_curve)
            
            # Performance drop at worst SNR
            worst_acc_drop = accuracy_curve[0] - min(accuracy_curve[1:])
            worst_f1_drop = f1_curve[0] - min(f1_curve[1:])
            
            degradation_analysis['robustness_metrics'] = {
                'auc_accuracy': auc_accuracy,
                'auc_f1_macro': auc_f1,
                'max_accuracy_drop': worst_acc_drop,
                'max_f1_drop': worst_f1_drop
            }
        
        return degradation_analysis
    
    def find_representative_speaker(
        self,
        test_df: pd.DataFrame,
        overall_accuracy_trend: np.ndarray,
        noisy_datasets: Dict[float, Tuple[np.ndarray, np.ndarray]],
        model
    ) -> int:
        """
        Find speaker most representative of overall degradation pattern.
        
        Args:
            test_df: Test dataset DataFrame
            overall_accuracy_trend: Overall accuracy trend across SNRs
            noisy_datasets: Noisy datasets per SNR level
            model: Trained model
            
        Returns:
            Speaker ID that best represents degradation pattern
        """
        logger.info("Finding representative speaker...")
        
        test_speaker_ids = sorted(test_df['speaker_id'].unique())
        speaker_id_array = test_df['speaker_id'].values
        
        # Calculate per-speaker accuracy for each SNR
        per_speaker_accuracies = {sp_id: [] for sp_id in test_speaker_ids}
        
        for snr in sorted(self.snr_levels):
            if snr not in noisy_datasets:
                continue
                
            X_noisy, y_true_noisy = noisy_datasets[snr]
            y_pred_noisy = model.predict(X_noisy)
            
            for speaker_id in test_speaker_ids:
                speaker_mask = (speaker_id_array == speaker_id)
                if np.sum(speaker_mask) > 0:
                    speaker_accuracy = accuracy_score(
                        y_true_noisy[speaker_mask], 
                        y_pred_noisy[speaker_mask]
                    )
                    per_speaker_accuracies[speaker_id].append(speaker_accuracy)
                else:
                    per_speaker_accuracies[speaker_id].append(np.nan)
        
        # Find speaker with minimum MSE to overall trend
        speaker_errors = {}
        for speaker_id, accuracy_trend in per_speaker_accuracies.items():
            speaker_trend_np = np.array(accuracy_trend)
            if not np.isnan(speaker_trend_np).any() and len(speaker_trend_np) == len(overall_accuracy_trend):
                mse = np.mean((speaker_trend_np - overall_accuracy_trend)**2)
                speaker_errors[speaker_id] = mse
        
        if speaker_errors:
            best_speaker_id = min(speaker_errors, key=speaker_errors.get)
            logger.info(f"Representative speaker: {best_speaker_id} (MSE: {speaker_errors[best_speaker_id]:.4f})")
            return best_speaker_id
        else:
            logger.warning("Could not find representative speaker")
            return test_speaker_ids[0]
    
    def statistical_significance_test(
        self,
        clean_results: np.ndarray,
        noisy_results: np.ndarray,
        test_type: str = 'paired_ttest'
    ) -> Dict:
        """
        Test statistical significance of performance differences.
        
        Args:
            clean_results: Performance on clean data
            noisy_results: Performance on noisy data
            test_type: Type of statistical test
            
        Returns:
            Statistical test results
        """
        if test_type == 'paired_ttest':
            statistic, p_value = stats.ttest_rel(clean_results, noisy_results)
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(clean_results, noisy_results)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        return {
            'test_type': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': np.mean(clean_results - noisy_results)
        }


class ModelComparisonAnalyzer:
    """Analyzer for comparing multiple models or conditions."""
    
    def __init__(self, config: Dict):
        """Initialize comparison analyzer."""
        self.config = config
        self.metrics_calculator = SERMetricsCalculator(config)
    
    def compare_models(
        self,
        model_results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Compare multiple models across metrics.
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            for metric_name, metric_data in results.items():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    comparison_data.append({
                        'model': model_name,
                        'metric': metric_name,
                        'mean': metric_data['mean'],
                        'lower_ci': metric_data.get('lower_ci', np.nan),
                        'upper_ci': metric_data.get('upper_ci', np.nan)
                    })
        
        return pd.DataFrame(comparison_data)
    
    def rank_models(
        self,
        comparison_df: pd.DataFrame,
        primary_metric: str = 'accuracy'
    ) -> pd.DataFrame:
        """
        Rank models by primary metric.
        
        Args:
            comparison_df: Model comparison DataFrame
            primary_metric: Primary metric for ranking
            
        Returns:
            Ranked models DataFrame
        """
        primary_results = comparison_df[comparison_df['metric'] == primary_metric].copy()
        primary_results = primary_results.sort_values('mean', ascending=False)
        primary_results['rank'] = range(1, len(primary_results) + 1)
        
        return primary_results[['model', 'mean', 'lower_ci', 'upper_ci', 'rank']]


class PerformanceReporter:
    """Comprehensive performance reporting system."""
    
    def __init__(self, config: Dict):
        """Initialize performance reporter."""
        self.config = config
        self.results_dir = config['paths']['results_dir']
    
    def generate_summary_report(
        self,
        clean_results: Dict,
        noisy_results: Dict[float, Dict],
        model_info: Dict,
        save_path: str = None
    ) -> str:
        """
        Generate comprehensive summary report.
        
        Args:
            clean_results: Clean data results
            noisy_results: Noisy data results
            model_info: Model information
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report_lines = [
            "# Speech Emotion Recognition Performance Report",
            "",
            "## Model Information",
            f"- Model Type: {model_info.get('model_type', 'XGBoost')}",
            f"- Training Samples: {model_info.get('training_samples', 'N/A')}",
            f"- Features: {model_info.get('num_features', 'N/A')}",
            f"- Emotions: {model_info.get('emotions', 'N/A')}",
            "",
            "## Clean Data Performance",
            f"- Accuracy: {clean_results['accuracy']['mean']:.4f} "
            f"(95% CI: {clean_results['accuracy']['lower_ci']:.4f} - "
            f"{clean_results['accuracy']['upper_ci']:.4f})",
            f"- F1-Score (Macro): {clean_results['f1_macro']['mean']:.4f} "
            f"(95% CI: {clean_results['f1_macro']['lower_ci']:.4f} - "
            f"{clean_results['f1_macro']['upper_ci']:.4f})",
            "",
            "## Noisy Data Performance",
        ]
        
        for snr in sorted(noisy_results.keys(), reverse=True):
            results = noisy_results[snr]
            report_lines.extend([
                f"### SNR {snr} dB",
                f"- Accuracy: {results['accuracy']['mean']:.4f} "
                f"(95% CI: {results['accuracy']['lower_ci']:.4f} - "
                f"{results['accuracy']['upper_ci']:.4f})",
                f"- F1-Score (Macro): {results['f1_macro']['mean']:.4f} "
                f"(95% CI: {results['f1_macro']['lower_ci']:.4f} - "
                f"{results['f1_macro']['upper_ci']:.4f})",
                ""
            ])
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")
        
        return report_text
    
    def save_results_to_files(
        self,
        results: Dict,
        prefix: str = "experiment"
    ) -> None:
        """
        Save results to various file formats.
        
        Args:
            results: Results dictionary
            prefix: File prefix
        """
        import pickle
        from pathlib import Path
        
        results_path = Path(self.results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle
        pickle_path = results_path / f"{prefix}_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save metrics as CSV if possible
        try:
            if 'metrics_df' in results:
                csv_path = results_path / f"{prefix}_metrics.csv"
                results['metrics_df'].to_csv(csv_path, index=False)
        except Exception as e:
            logger.warning(f"Could not save CSV: {e}")
        
        logger.info(f"Results saved to {results_path}")