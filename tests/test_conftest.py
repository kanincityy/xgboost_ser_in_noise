import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def temp_project_dir():
    """Create temporary project directory structure."""
    temp_dir = tempfile.mkdtemp()
    project_path = Path(temp_dir) / "test_project"
    
    # Create directory structure
    directories = [
        "data/raw", "data/processed", "data/models", "data/results/figures",
        "logs", "config"
    ]
    
    for directory in directories:
        (project_path / directory).mkdir(parents=True, exist_ok=True)
    
    yield project_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Standard configuration for testing."""
    return {
        'data': {
            'target_sr': 16000,
            'emotions': ['angry', 'calm', 'happy', 'sad']
        },
        'audio': {
            'n_fft': 1024,
            'hop_length': 512,
            'n_mels': 128,
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
        },
        'noise': {
            'snr_levels': [20, 10, 0, -10],
            'noise_type': 'white'
        },
        'model': {
            'xgboost': {
                'objective': 'multi:softmax',
                'num_class': 4,
                'tree_method': 'hist',
                'device': 'cpu',
                'random_state': 42,
                'n_estimators': 10
            }
        },
        'hyperopt': {
            'n_iter': 2,
            'cv_folds': 2,
            'random_state': 42
        },
        'training': {
            'test_size': 0.3,
            'val_split': 0.5,
            'random_state': 42,
            'speaker_independent': True
        },
        'evaluation': {
            'confidence_level': 95,
            'n_bootstraps': 100,  # Reduced for testing
            'metrics': ['accuracy', 'f1_score']
        },
        'visualization': {
            'figure_size': [10, 8],
            'dpi': 150,  # Reduced for testing
            'format': 'png',
            'top_n_features': 10
        },
        'paths': {
            'data_dir': 'data/',
            'results_dir': 'data/results/',
            'models_dir': 'data/models/',
            'figures_dir': 'data/results/figures/'
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    def _generate_audio(duration=2.0, sr=16000, frequency=440):
        t = np.linspace(0, duration, int(sr * duration), False)
        # Create a more complex signal with harmonics
        signal = (
            np.sin(frequency * 2 * np.pi * t) +
            0.3 * np.sin(frequency * 2 * 2 * np.pi * t) +
            0.1 * np.random.randn(len(t))
        )
        return signal.astype(np.float32), sr
    
    return _generate_audio


@pytest.fixture
def sample_features_dataframe():
    """Generate sample features DataFrame."""
    def _generate_features_df(n_samples=100, n_features=50, emotions=None):
        if emotions is None:
            emotions = ['angry', 'calm', 'happy', 'sad']
        
        # Generate synthetic features
        features = np.random.randn(n_samples, n_features)
        
        # Make features somewhat realistic (positive for energy, etc.)
        features[:, :5] = np.abs(features[:, :5])  # Energy features
        
        # Create feature names
        feature_names = []
        feature_names.extend(['meanF0', 'stdF0', 'medianF0', 'minF0', 'maxF0'])
        feature_names.extend(['meanEnergy', 'stdEnergy', 'medianEnergy', 'minEnergy', 'maxEnergy'])
        feature_names.extend([f'mfcc{i}_mean' for i in range(13)])
        feature_names.extend([f'mfcc{i}_std' for i in range(13)])
        feature_names.extend([f'delta_mfcc{i}_mean' for i in range(13)])
        feature_names.extend([f'delta_mfcc{i}_std' for i in range(4)])  # Truncated for exact n_features
        
        feature_names = feature_names[:n_features]  # Ensure exact number
        
        # Generate labels
        labels = np.random.choice(emotions, size=n_samples)
        
        # Create DataFrame
        import pandas as pd
        features_df = pd.DataFrame(features, columns=feature_names)
        
        return features_df, labels, feature_names
    
    return _generate_features_df


@pytest.fixture
def mock_audio_files(tmp_path):
    """Create mock audio files for testing."""
    audio_files = []
    emotions = ['angry', 'calm', 'happy', 'sad']
    
    for speaker_id in [1, 2]:
        speaker_dir = tmp_path / f"Actor_{speaker_id:02d}"
        speaker_dir.mkdir()
        
        for emotion_idx, emotion in enumerate(emotions, 1):
            for intensity in [1, 2]:
                for statement in [1, 2]:
                    filename = f"03-01-{emotion_idx:02d}-{intensity:02d}-{statement:02d}-01-{speaker_id:02d}.wav"
                    file_path = speaker_dir / filename
                    
                    # Create empty file (in real tests, would contain audio)
                    file_path.touch()
                    
                    audio_files.append({
                        'full_path': str(file_path),
                        'emotion_label': emotion,
                        'speaker_id': speaker_id,
                        'filename': filename
                    })
    
    return audio_files