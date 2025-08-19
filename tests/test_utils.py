import pytest
import tempfile
import yaml
from pathlib import Path

from utils.helpers import (
    load_config, save_experiment_config, set_random_seeds,
    create_directory_structure, validate_config
)


class TestHelperUtils:
    """Test cases for helper utilities."""
    
    def test_load_config(self):
        """Test configuration loading."""
        config_data = {
            'data': {'target_sr': 16000, 'emotions': ['happy', 'sad']},
            'paths': {'data_dir': 'data/', 'results_dir': 'results/'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            loaded_config = load_config(config_path)
            assert loaded_config['data']['target_sr'] == 16000
            assert len(loaded_config['data']['emotions']) == 2
        finally:
            Path(config_path).unlink()
    
    def test_save_experiment_config(self):
        """Test experiment configuration saving."""
        config = {'test': 'value'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / 'test_config.yaml'
            save_experiment_config(config, str(save_path))
            
            assert save_path.exists()
            
            # Load and verify
            with open(save_path, 'r') as f:
                loaded = yaml.safe_load(f)
            
            assert loaded['test'] == 'value'
            assert 'experiment_metadata' in loaded
    
    def test_set_random_seeds(self):
        """Test random seed setting."""
        import numpy as np
        import random
        
        set_random_seeds(42)
        
        # Test that seeds are actually set
        val1 = np.random.rand()
        rand1 = random.random()
        
        set_random_seeds(42)  # Reset with same seed
        
        val2 = np.random.rand()
        rand2 = random.random()
        
        assert val1 == val2
        assert rand1 == rand2
    
    def test_create_directory_structure(self):
        """Test directory structure creation."""
        config = {'paths': {}}  # Minimal config
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dirs = create_directory_structure(temp_dir, config)
            
            assert 'data' in dirs
            assert 'models' in dirs
            assert 'results' in dirs
            
            # Verify directories exist
            for dir_path in dirs.values():
                assert Path(dir_path).exists()
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            'data': {'emotions': ['happy', 'sad'], 'target_sr': 16000},
            'audio': {}, 'features': {}, 'model': {}, 'training': {},
            'evaluation': {}, 'paths': {}, 'noise': {'snr_levels': [10, 0]}
        }
        
        assert validate_config(valid_config) == True
        
        # Invalid config - missing required key
        invalid_config = {'data': {}}
        
        with pytest.raises(ValueError, match="Missing required configuration key"):
            validate_config(invalid_config)
        
        # Invalid config - missing emotions
        invalid_config2 = {
            'data': {'target_sr': 16000},
            'audio': {}, 'features': {}, 'model': {}, 'training': {},
            'evaluation': {}, 'paths': {}, 'noise': {'snr_levels': [10]}
        }
        
        with pytest.raises(ValueError, match="Missing 'emotions'"):
            validate_config(invalid_config2)