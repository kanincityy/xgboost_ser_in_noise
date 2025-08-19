import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from features.extractor import AudioFeatureExtractor
from sklearn.preprocessing import LabelEncoder


class TestAudioFeatureExtractor:
    """Test cases for AudioFeatureExtractor."""
    
    @pytest.fixture
    def config(self):
        """Sample configuration for testing."""
        return {
            'data': {'target_sr': 16000},
            'audio': {
                'n_fft': 1024,
                'hop_length': 512,
                'n_mfcc': 13
            },
            'features': {
                'include_f0': True,
                'include_energy': True,
                'include_mfcc': True,
                'include_delta_mfcc': True,
                'include_delta2_mfcc': True,
                'include_zcr': True,
                'include_spectral_centroid': True
            }
        }
    
    @pytest.fixture
    def extractor(self, config):
        """Create feature extractor instance."""
        return AudioFeatureExtractor(config)
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio signal."""
        duration = 2.0  # seconds
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration), False)
        # Create a simple sine wave with some noise
        signal = np.sin(440 * 2 * np.pi * t) + 0.1 * np.random.randn(len(t))
        return signal.astype(np.float32), sr
    
    def test_extractor_initialization(self, extractor, config):
        """Test extractor initialization."""
        assert extractor.target_sr == config['data']['target_sr']
        assert extractor.n_fft == config['audio']['n_fft']
        assert extractor.n_mfcc == config['audio']['n_mfcc']
    
    def test_f0_feature_extraction(self, extractor, sample_audio):
        """Test F0 feature extraction."""
        signal, sr = sample_audio
        features, feature_names = extractor.extract_f0_features(signal, sr)
        
        assert len(features) == 5  # meanF0, stdF0, medianF0, minF0, maxF0
        assert len(feature_names) == 5
        assert all(isinstance(f, (int, float)) for f in features)
        assert 'meanF0' in feature_names
    
    def test_energy_feature_extraction(self, extractor, sample_audio):
        """Test energy feature extraction.""" 
        signal, sr = sample_audio
        features, feature_names = extractor.extract_energy_features(signal)
        
        assert len(features) == 5  # meanEnergy, stdEnergy, etc.
        assert len(feature_names) == 5
        assert all(f >= 0 for f in features)  # Energy should be non-negative
        assert 'meanEnergy' in feature_names
    
    def test_mfcc_feature_extraction(self, extractor, sample_audio):
        """Test MFCC feature extraction."""
        signal, sr = sample_audio
        features, feature_names = extractor.extract_mfcc_features(signal, sr)
        
        expected_size = extractor.n_mfcc * 2 * 3  # base + delta + delta2, each with mean + std
        assert len(features) == expected_size
        assert len(feature_names) == expected_size
        assert any('mfcc0_mean' in name for name in feature_names)
    
    def test_complete_feature_extraction(self, extractor, sample_audio):
        """Test complete feature extraction pipeline."""
        signal, sr = sample_audio
        features, feature_names = extractor.extract_features_from_signal(signal, sr)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert len(feature_names) == len(features)
        assert not np.all(np.isnan(features))
    
    @patch('librosa.load')
    def test_process_dataframe_single_file(self, mock_load, extractor, sample_audio):
        """Test processing a single file through DataFrame."""
        signal, sr = sample_audio
        mock_load.return_value = (signal, sr)
        
        # Create sample DataFrame
        df = pd.DataFrame({
            'full_path': ['/fake/path/audio.wav'],
            'emotion_label': ['happy']
        })
        
        features_df, labels, le, feature_names, means = extractor.process_dataframe(
            df, fit_label_encoder=True
        )
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 1
        assert isinstance(le, LabelEncoder)
        assert len(feature_names) > 0
        mock_load.assert_called_once()
    
    def test_feature_size_calculation(self, extractor):
        """Test expected feature size calculation."""
        expected_size = extractor._calculate_expected_feature_size()
        assert expected_size > 0
        assert isinstance(expected_size, int)
    
    def test_error_handling_invalid_audio(self, extractor):
        """Test error handling with invalid audio."""
        # Test with empty signal
        empty_signal = np.array([])
        features, feature_names = extractor.extract_features_from_signal(empty_signal, 16000)
        
        # Should return NaN array when extraction fails
        assert np.all(np.isnan(features))
