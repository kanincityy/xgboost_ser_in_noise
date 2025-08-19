#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path
import yaml
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.dataset_handler import RAVDESSHandler
from features.extractor import AudioFeatureExtractor
from models.xgboost_model import XGBoostSERModel
from utils.noise_generator import NoiseGenerator
from evaluation.metrics import SERMetricsCalculator, NoiseRobustnessAnalyzer
from evaluation.visualization import SERVisualizer
from utils.helpers import setup_logging, save_experiment_config

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main experimental pipeline."""
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition in White Noise')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--download', action='store_true',
                       help='Download dataset (default: False)')
    parser.add_argument('--optimize', action='store_true', default=True,
                       help='Optimize hyperparameters (default: True)')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualizations (default: True)')
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='Save trained model (default: True)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    logger.info("Starting Speech Emotion Recognition experiment")
    logger.info(f"Configuration: {args.config}")
    
    # Save experiment configuration
    save_experiment_config(config, Path(config['paths']['results_dir']) / 'experiment_config.yaml')
    
    try:
        # Initialize components
        dataset_handler = RAVDESSHandler(config)
        feature_extractor = AudioFeatureExtractor(config)
        model = XGBoostSERModel(config)
        noise_generator = NoiseGenerator(config)
        metrics_calculator = SERMetricsCalculator(config)
        robustness_analyzer = NoiseRobustnessAnalyzer(config)
        
        # Step 1: Dataset preparation
        logger.info("=" * 60)
        logger.info("STEP 1: Dataset Preparation")
        logger.info("=" * 60)
        
        train_df, val_df, test_df = dataset_handler.prepare_dataset(download=args.download)
        
        # Step 2: Feature extraction
        logger.info("=" * 60)
        logger.info("STEP 2: Feature Extraction")
        logger.info("=" * 60)
        
        # Extract features for training set
        train_features_df, train_labels, label_encoder, feature_names, feature_means = \
            feature_extractor.process_dataframe(train_df, fit_label_encoder=True)
        
        # Extract features for validation set
        val_features_df, val_labels, _, _, _ = \
            feature_extractor.process_dataframe(
                val_df, 
                label_encoder=label_encoder, 
                feature_means=feature_means,
                feature_names=feature_names
            )
        
        # Extract features for test set (clean)
        test_features_df, test_labels, _, _, _ = \
            feature_extractor.process_dataframe(
                test_df,
                label_encoder=label_encoder,
                feature_means=feature_means,
                feature_names=feature_names
            )
        
        logger.info(f"Training features: {train_features_df.shape}")
        logger.info(f"Validation features: {val_features_df.shape}")
        logger.info(f"Test features: {test_features_df.shape}")
        
        # Step 3: Model training
        logger.info("=" * 60)
        logger.info("STEP 3: Model Training")
        logger.info("=" * 60)
        
        # Prepare training pool for hyperparameter optimization
        training_pool_df = pd.concat([train_df, val_df], ignore_index=True)
        training_pool_groups = training_pool_df['speaker_id'].values
        
        # Train model
        model.train(
            train_features_df=train_features_df,
            train_labels=train_labels,
            val_features_df=val_features_df,
            val_labels=val_labels,
            feature_names=feature_names,
            groups=training_pool_groups,
            optimize_hyperparams=args.optimize
        )
        
        logger.info("Model training completed")
        if model.best_params:
            logger.info(f"Best parameters: {model.best_params}")
        
        # Step 4: Clean data evaluation
        logger.info("=" * 60)
        logger.info("STEP 4: Clean Data Evaluation")
        logger.info("=" * 60)
        
        # Evaluate on clean test data
        test_features_scaled, test_labels_encoded = model.process_noisy_data(
            test_features_df, test_labels
        )
        
        clean_results = model.evaluate(test_features_scaled, test_labels_encoded)
        logger.info(f"Clean test accuracy: {clean_results['accuracy']:.4f}")
        
        # Calculate metrics with confidence intervals
        clean_metrics_with_ci = metrics_calculator.calculate_overall_metrics_with_ci(
            test_labels_encoded, clean_results['predictions']
        )
        
        per_emotion_clean = metrics_calculator.calculate_per_emotion_metrics(
            test_labels_encoded, clean_results['predictions'], 
            list(label_encoder.classes_)
        )
        
        # Step 5: Noise robustness evaluation
        logger.info("=" * 60)
        logger.info("STEP 5: Noise Robustness Evaluation")
        logger.info("=" * 60)
        
        # Process noisy test data
        noisy_datasets = noise_generator.process_dataset_with_noise(
            test_df, feature_extractor, config['data']['target_sr']
        )
        
        # Evaluate on noisy data
        noisy_results = {}
        per_emotion_noisy = {}
        
        for snr_db, (noisy_features, noisy_labels_str) in noisy_datasets.items():
            logger.info(f"Evaluating SNR {snr_db} dB...")
            
            # Convert features to DataFrame and process
            noisy_features_df = pd.DataFrame(noisy_features, columns=feature_names)
            noisy_features_scaled, noisy_labels_encoded = model.process_noisy_data(
                noisy_features_df, noisy_labels_str
            )
            
            # Evaluate
            snr_results = model.evaluate(noisy_features_scaled, noisy_labels_encoded)
            logger.info(f"SNR {snr_db} dB accuracy: {snr_results['accuracy']:.4f}")
            
            # Calculate metrics with CI
            noisy_metrics_with_ci = metrics_calculator.calculate_overall_metrics_with_ci(
                noisy_labels_encoded, snr_results['predictions']
            )
            noisy_results[snr_db] = noisy_metrics_with_ci
            
            # Per-emotion metrics
            per_emotion_noisy[snr_db] = metrics_calculator.calculate_per_emotion_metrics(
                noisy_labels_encoded, snr_results['predictions'],
                list(label_encoder.classes_)
            )
        
        # Step 6: Robustness analysis
        logger.info("=" * 60)
        logger.info("STEP 6: Robustness Analysis")
        logger.info("=" * 60)
        
        degradation_analysis = robustness_analyzer.analyze_degradation_patterns(
            clean_metrics_with_ci, noisy_results
        )
        
        # Find representative speaker
        overall_accuracy_trend = [noisy_results[snr]['accuracy']['mean'] 
                                 for snr in sorted(config['noise']['snr_levels'])]
        representative_speaker = robustness_analyzer.find_representative_speaker(
            test_df, np.array(overall_accuracy_trend), noisy_datasets, model
        )
        
        logger.info(f"Representative speaker: {representative_speaker}")
        
        # Step 7: Visualization and reporting
        if args.visualize:
            logger.info("=" * 60)
            logger.info("STEP 7: Visualization and Reporting")
            logger.info("=" * 60)
            
            visualizer = SERVisualizer(config)
            
            # Plot performance degradation
            visualizer.plot_performance_degradation(
                clean_metrics_with_ci, noisy_results, 
                config['noise']['snr_levels'],
                config['data']['emotions']
            )
            
            # Plot per-emotion performance
            all_per_emotion = {'Clean': per_emotion_clean}
            all_per_emotion.update(per_emotion_noisy)
            visualizer.plot_per_emotion_performance(
                all_per_emotion, config['noise']['snr_levels']
            )
            
            # Plot confusion matrices
            all_confusion_matrices = {'Clean': clean_results['confusion_matrix']}
            for snr_db, (_, _) in noisy_datasets.items():
                # Get confusion matrix for this SNR
                noisy_features_df = pd.DataFrame(noisy_datasets[snr_db][0], columns=feature_names)
                noisy_features_scaled, noisy_labels_encoded = model.process_noisy_data(
                    noisy_features_df, noisy_datasets[snr_db][1]
                )
                snr_eval = model.evaluate(noisy_features_scaled, noisy_labels_encoded)
                all_confusion_matrices[f'SNR {snr_db}dB'] = snr_eval['confusion_matrix']
            
            visualizer.plot_confusion_matrices(
                all_confusion_matrices, list(label_encoder.classes_)
            )
            
            # Plot feature importance
            feature_importance_df = model.get_feature_importance()
            visualizer.plot_feature_importance(feature_importance_df)
            
            # Create degradation sequence for representative speaker
            representative_files = test_df[test_df['speaker_id'] == representative_speaker]
            for emotion in config['data']['emotions']:
                emotion_files = representative_files[representative_files['emotion_label'] == emotion]
                if not emotion_files.empty:
                    clean_audio_path = emotion_files.iloc[0]['full_path']
                    visualizer.create_degradation_sequence(
                        clean_audio_path, [20, 5, -5], emotion, 
                        representative_speaker, noise_generator
                    )
        
        # Step 8: Save results and model
        logger.info("=" * 60)
        logger.info("STEP 8: Saving Results")
        logger.info("=" * 60)
        
        # Save model
        if args.save_model:
            model_save_dir = Path(config['paths']['models_dir'])
            model.save_model(str(model_save_dir))
            logger.info(f"Model saved to {model_save_dir}")
        
        # Save experimental results
        from evaluation.metrics import PerformanceReporter
        reporter = PerformanceReporter(config)
        
        # Generate comprehensive report
        model_info = {
            'model_type': 'XGBoost',
            'training_samples': len(train_features_df) + len(val_features_df),
            'num_features': len(feature_names),
            'emotions': config['data']['emotions']
        }
        
        report_text = reporter.generate_summary_report(
            clean_metrics_with_ci, noisy_results, model_info,
            save_path=str(Path(config['paths']['results_dir']) / 'experiment_report.md')
        )
        
        # Save detailed results
        detailed_results = {
            'config': config,
            'clean_results': clean_metrics_with_ci,
            'noisy_results': noisy_results,
            'degradation_analysis': degradation_analysis,
            'feature_importance': feature_importance_df.to_dict(),
            'representative_speaker': representative_speaker,
            'model_info': model_info
        }
        
        reporter.save_results_to_files(detailed_results, 'comprehensive_experiment')
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to: {config['paths']['results_dir']}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Clean test accuracy: {clean_metrics_with_ci['accuracy']['mean']:.4f} "
              f"(95% CI: {clean_metrics_with_ci['accuracy']['lower_ci']:.4f} - "
              f"{clean_metrics_with_ci['accuracy']['upper_ci']:.4f})")
        
        print("\nNoisy test performance:")
        for snr in sorted(noisy_results.keys(), reverse=True):
            acc = noisy_results[snr]['accuracy']
            print(f"  SNR {snr:3} dB: {acc['mean']:.4f} "
                  f"(95% CI: {acc['lower_ci']:.4f} - {acc['upper_ci']:.4f})")
        
        print(f"\nRepresentative speaker: {representative_speaker}")
        print(f"Results directory: {config['paths']['results_dir']}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()


# =============================================================================
# Additional utility scripts
# =============================================================================

def create_train_script():
    """Create dedicated training script."""
    train_script = '''#!/usr/bin/env python3
"""
Training script for XGBoost SER model.

Usage:
    python scripts/train_model.py --config config/config.yaml --data-dir data/processed/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path  
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.xgboost_model import XGBoostSERModel
from utils.helpers import load_config, setup_logging

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost SER model')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--data-dir', required=True, help='Processed data directory')
    parser.add_argument('--output-dir', required=True, help='Model output directory')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logging(config)
    
    # Load processed data
    train_features = pd.read_csv(Path(args.data_dir) / 'train_features.csv')
    # ... implement training logic
    
if __name__ == "__main__":
    main()
'''
    return train_script


def create_evaluate_script():
    """Create dedicated evaluation script."""
    eval_script = '''#!/usr/bin/env python3
"""
Evaluation script for trained SER model.

Usage:
    python scripts/evaluate_model.py --model-dir data/models/ --test-data data/processed/test/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.xgboost_model import XGBoostSERModel
from evaluation.metrics import SERMetricsCalculator
from evaluation.visualization import SERVisualizer
from utils.helpers import load_config, setup_logging

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained SER model')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--model-dir', required=True, help='Trained model directory')
    parser.add_argument('--test-data', required=True, help='Test data directory')
    parser.add_argument('--output-dir', default='results/evaluation/', help='Output directory')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logging(config)
    
    # Load model
    model = XGBoostSERModel(config)
    model.load_model(args.model_dir)
    
    # Load test data
    test_features = pd.read_csv(Path(args.test_data) / 'test_features.csv')
    test_labels = np.load(Path(args.test_data) / 'test_labels.npy')
    
    # Evaluate model
    results = model.evaluate(test_features.values, test_labels)
    
    # Generate visualizations
    visualizer = SERVisualizer(config)
    # ... implement evaluation logic
    
if __name__ == "__main__":
    main()
'''
    return eval_script