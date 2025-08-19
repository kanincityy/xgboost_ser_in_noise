import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from data.dataset_handler import RAVDESSHandler
from features.extractor import AudioFeatureExtractor
from models.xgboost_model import XGBoostSERModel
from utils.noise_generator import NoiseGenerator


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete SER pipeline."""
    
    def test_dataset_to_features_pipeline(self, sample_config, mock_audio_files, sample_audio_data):
        """Test pipeline from dataset loading to feature extraction."""
        # Mock dataset handler
        with patch.object(RAVDESSHandler, 'download_dataset'):
            with patch.object(RAVDESSHandler, 'load_metadata') as mock_load:
                # Create mock metadata
                mock_df = pd.DataFrame(mock_audio_files)
                mock_load.return_value = mock_df
                
                handler = RAVDESSHandler(sample_config)
                train_df, val_df, test_df = handler.prepare_dataset(download=False)
                
                # Verify splits
                assert len(train_df) > 0
                assert len(val_df) > 0
                assert len(test_df) > 0
                
                # Verify no speaker overlap
                train_speakers = set(train_df['speaker_id'].unique())
                test_speakers = set(test_df['speaker_id'].unique())
                assert len(train_speakers.intersection(test_speakers)) == 0
    
    @patch('librosa.load')
    def test_features_to_model_pipeline(self, mock_load, sample_config, sample_features_dataframe):
        """Test pipeline from feature extraction to model training."""
        # Mock librosa.load
        audio_data, sr = sample_features_dataframe(n_samples=1)  # Get sample audio function
        mock_load.return_value = (np.random.randn(16000), 16000)
        
        # Create mock dataframes
        features_df, labels, feature_names = sample_features_dataframe(n_samples=50, n_features=20)
        
        # Mock dataset splits
        split_idx = len(features_df) // 2
        train_df = pd.DataFrame({
            'full_path': ['/fake/path.wav'] * split_idx,
            'emotion_label': labels[:split_idx],
            'speaker_id': list(range(1, split_idx + 1))
        })
        
        val_df = pd.DataFrame({
            'full_path': ['/fake/path.wav'] * (len(features_df) - split_idx),
            'emotion_label': labels[split_idx:],
            'speaker_id': list(range(split_idx + 1, len(features_df) + 1))
        })
        
        # Test feature extraction
        extractor = AudioFeatureExtractor(sample_config)
        
        # Mock the feature extraction to return our sample data
        with patch.object(extractor, 'extract_features_from_signal') as mock_extract:
            mock_extract.return_value = (features_df.iloc[0].values, feature_names)
            
            train_features_df, train_labels_enc, le, feat_names, means = extractor.process_dataframe(
                train_df, fit_label_encoder=True
            )
            
            val_features_df, val_labels_enc, _, _, _ = extractor.process_dataframe(
                val_df, label_encoder=le, feature_means=means, feature_names=feat_names
            )
            
            # Test model training
            model = XGBoostSERModel(sample_config)
            groups = np.arange(len(train_features_df) + len(val_features_df))
            
            # Train without hyperparameter optimization for speed
            model.train(
                train_features_df, train_labels_enc,
                val_features_df, val_labels_enc,
                feat_names, groups, optimize_hyperparams=False
            )
            
            assert model.is_fitted
            assert model.model is not None
    
    def test_noise_robustness_pipeline(self, sample_config, sample_features_dataframe):
        """Test noise robustness evaluation pipeline."""
        # Create sample data
        features_df, labels, feature_names = sample_features_dataframe(n_samples=20, n_features=10)
        
        # Quick model training
        model = XGBoostSERModel(sample_config)
        split_idx = len(features_df) // 2
        
        train_features = features_df[:split_idx]
        train_labels = labels[:split_idx]
        val_features = features_df[split_idx:]
        val_labels = labels[split_idx:]
        groups = np.arange(len(features_df))
        
        model.train(
            train_features, train_labels, val_features, val_labels,
            feature_names, groups, optimize_hyperparams=False
        )
        
        # Test noise generation and evaluation
        noise_generator = NoiseGenerator(sample_config)
        
        # Mock test dataframe
        test_df = pd.DataFrame({
            'full_path': ['/fake/test.wav'] * 10,
            'emotion_label': labels[:10],
            'speaker_id': list(range(1, 11))
        })
        
        # Mock noise dataset processing
        with patch.object(noise_generator, 'process_dataset_with_noise') as mock_process:
            # Mock return data for different SNR levels
            mock_noisy_data = {}
            for snr in sample_config['noise']['snr_levels']:
                mock_features = np.random.randn(10, len(feature_names))
                mock_labels = labels[:10]
                mock_noisy_data[snr] = (mock_features, mock_labels)
            
            mock_process.return_value = mock_noisy_data
            
            noisy_datasets = noise_generator.process_dataset_with_noise(
                test_df, None, sample_config['data']['target_sr']
            )
            
            # Test evaluation on noisy data
            results = {}
            for snr_db, (noisy_features, noisy_labels_str) in noisy_datasets.items():
                # Convert to DataFrame and process
                noisy_features_df = pd.DataFrame(noisy_features, columns=feature_names)
                noisy_features_scaled, noisy_labels_encoded = model.process_noisy_data(
                    noisy_features_df, noisy_labels_str
                )
                
                # Evaluate
                eval_results = model.evaluate(noisy_features_scaled, noisy_labels_encoded)
                results[snr_db] = eval_results['accuracy']
            
            # Verify we have results for all SNR levels
            assert len(results) == len(sample_config['noise']['snr_levels'])
            assert all(0 <= acc <= 1 for acc in results.values())


@pytest.mark.slow
class TestPerformanceTests:
    """Performance and stress tests."""
    
    def test_large_dataset_processing(self, sample_config):
        """Test processing with larger datasets."""
        # This test would be run with @pytest.mark.slow
        # and only when specifically requested
        pass
    
    def test_memory_usage(self, sample_config):
        """Test memory usage during processing."""
        # Monitor memory usage during feature extraction
        pass


# Run with: pytest tests/ -v
# Run integration tests: pytest tests/integration/ -v -m integration
# Run without slow tests: pytest tests/ -v -m "not slow"