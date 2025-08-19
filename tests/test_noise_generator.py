import pytest
import numpy as np
from unittest.mock import patch

from utils.noise_generator import NoiseGenerator


class TestNoiseGenerator:
    """Test cases for NoiseGenerator."""
    
    @pytest.fixture
    def config(self):
        """Sample configuration."""
        return {
            'noise': {
                'snr_levels': [20, 10, 0, -10],
                'noise_type': 'white'
            }
        }
    
    @pytest.fixture
    def generator(self, config):
        """Create noise generator instance."""
        return NoiseGenerator(config)
    
    @pytest.fixture
    def clean_signal(self):
        """Create clean test signal."""
        duration = 1.0
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration), False)
        return np.sin(440 * 2 * np.pi * t).astype(np.float32)
    
    def test_generator_initialization(self, generator, config):
        """Test generator initialization."""
        assert generator.snr_levels == config['noise']['snr_levels']
        assert generator.noise_type == config['noise']['noise_type']
    
    def test_white_noise_addition(self, generator, clean_signal):
        """Test white noise addition."""
        target_snr = 10  # dB
        noisy_signal = generator.add_white_noise(clean_signal, target_snr)
        
        assert len(noisy_signal) == len(clean_signal)
        assert noisy_signal.dtype == np.float32
        assert not np.array_equal(clean_signal, noisy_signal)
    
    def test_snr_calculation_accuracy(self, generator, clean_signal):
        """Test SNR calculation accuracy."""
        target_snr = 10  # dB
        noisy_signal = generator.add_white_noise(clean_signal, target_snr)
        actual_snr = generator.calculate_actual_snr(clean_signal, noisy_signal)
        
        # Allow 1dB tolerance due to random noise
        assert abs(actual_snr - target_snr) < 1.0
    
    def test_multiple_snr_levels(self, generator, clean_signal):
        """Test noise addition at multiple SNR levels."""
        snr_levels = [20, 10, 0, -10]
        
        for snr in snr_levels:
            noisy_signal = generator.add_noise(clean_signal, snr)
            assert len(noisy_signal) == len(clean_signal)
            
            # Higher SNR should result in less noise
            noise = noisy_signal - clean_signal
            noise_power = np.mean(noise**2)
            
            # Verify noise power increases as SNR decreases
            if snr == 20:
                high_snr_noise_power = noise_power
            elif snr == -10:
                low_snr_noise_power = noise_power
                assert low_snr_noise_power > high_snr_noise_power
    
    def test_edge_cases(self, generator):
        """Test edge cases."""
        # Test with silent signal
        silent_signal = np.zeros(1000)
        noisy_signal = generator.add_white_noise(silent_signal, 10)
        assert len(noisy_signal) == len(silent_signal)
        
        # Test with very short signal
        short_signal = np.array([1.0, 2.0, 3.0])
        noisy_short = generator.add_white_noise(short_signal, 0)
        assert len(noisy_short) == len(short_signal)
    
    def test_pink_noise_addition(self, generator, clean_signal):
        """Test pink noise addition."""
        noisy_signal = generator.add_pink_noise(clean_signal, 10)
        
        assert len(noisy_signal) == len(clean_signal)
        assert not np.array_equal(clean_signal, noisy_signal)
    
    def test_snr_validation(self, generator, clean_signal):
        """Test SNR validation with multiple samples."""
        test_signals = [clean_signal] * 5
        validation_results = generator.validate_snr_levels(test_signals, num_samples=3)
        
        assert isinstance(validation_results, dict)
        for snr in generator.snr_levels:
            if snr in validation_results:
                assert 'mean' in validation_results[snr]
                assert 'std' in validation_results[snr]