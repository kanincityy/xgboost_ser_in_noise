import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import random
import os
import json
from datetime import datetime


def setup_logging(config: Dict, log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration dictionary
        log_file: Optional log file path
    """
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory
    if log_file is None:
        log_file = log_config.get('file', 'logs/ser_experiment.log')
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('librosa').setLevel(logging.WARNING)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand paths relative to config file location
    config_dir = config_path.parent
    for path_key in ['data_dir', 'results_dir', 'models_dir']:
        if path_key in config.get('paths', {}):
            path_value = config['paths'][path_key]
            if not os.path.isabs(path_value):
                config['paths'][path_key] = str(config_dir / path_value)
    
    return config


def save_experiment_config(config: Dict, save_path: str) -> None:
    """
    Save experiment configuration with timestamp.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add experiment metadata
    config_with_meta = config.copy()
    config_with_meta['experiment_metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'working_directory': str(Path.cwd())
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(config_with_meta, f, default_flow_style=False, indent=2)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set additional seeds for libraries
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def create_directory_structure(base_dir: str, config: Dict) -> Dict[str, str]:
    """
    Create project directory structure.
    
    Args:
        base_dir: Base directory path
        config: Configuration dictionary
        
    Returns:
        Dictionary of created directory paths
    """
    base_path = Path(base_dir)
    
    directories = {
        'data': base_path / 'data',
        'raw_data': base_path / 'data' / 'raw',
        'processed_data': base_path / 'data' / 'processed',
        'models': base_path / 'data' / 'models',
        'results': base_path / 'data' / 'results',
        'figures': base_path / 'data' / 'results' / 'figures',
        'logs': base_path / 'logs'
    }
    
    # Create directories
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
    
    return {name: str(path) for name, path in directories.items()}


def validate_config(config: Dict) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = [
        'data', 'audio', 'features', 'model', 'training', 'evaluation', 'paths'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate specific sections
    if 'emotions' not in config['data']:
        raise ValueError("Missing 'emotions' in data configuration")
    
    if 'target_sr' not in config['data']:
        raise ValueError("Missing 'target_sr' in data configuration")
    
    if 'snr_levels' not in config['noise']:
        raise ValueError("Missing 'snr_levels' in noise configuration")
    
    return True


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage information
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident set size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'error': 'psutil not available'}


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator  
        default: Default value if denominator is zero
        
    Returns:
        Division result or default value
    """
    return numerator / denominator if denominator != 0 else default


def ensure_dir_exists(path: str) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_audio_metadata(metadata_path: str) -> Dict:
    """
    Load audio metadata from various formats.
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Metadata dictionary
    """
    metadata_path = Path(metadata_path)
    
    if metadata_path.suffix.lower() == '.json':
        with open(metadata_path, 'r') as f:
            return json.load(f)
    elif metadata_path.suffix.lower() in ['.yml', '.yaml']:
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")


def calculate_dataset_statistics(features_df, labels) -> Dict:
    """
    Calculate dataset statistics.
    
    Args:
        features_df: Features DataFrame
        labels: Labels array
        
    Returns:
        Statistics dictionary
    """
    stats = {
        'n_samples': len(features_df),
        'n_features': features_df.shape[1],
        'n_classes': len(np.unique(labels)),
        'class_distribution': {},
        'feature_statistics': {
            'mean': features_df.mean().to_dict(),
            'std': features_df.std().to_dict(),
            'min': features_df.min().to_dict(),
            'max': features_df.max().to_dict()
        }
    }
    
    # Calculate class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        stats['class_distribution'][str(label)] = int(count)
    
    return stats


def export_results_summary(results: Dict, output_path: str) -> None:
    """
    Export results summary to multiple formats.
    
    Args:
        results: Results dictionary
        output_path: Output file path (without extension)
    """
    output_path = Path(output_path)
    
    # Export as JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = convert_numpy_to_lists(results)
        json.dump(json_results, f, indent=2, default=str)
    
    # Export as YAML
    yaml_path = output_path.with_suffix('.yaml')
    with open(yaml_path, 'w') as f:
        yaml_results = convert_numpy_to_lists(results)
        yaml.dump(yaml_results, f, default_flow_style=False, indent=2)
    
    logging.info(f"Results exported to {json_path} and {yaml_path}")


def convert_numpy_to_lists(obj):
    """
    Recursively convert numpy arrays to lists for serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) for item in obj]
    else:
        return obj


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            description: Description of the operation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, step: int = None, message: str = None):
        """Update progress."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        percentage = (self.current_step / self.total_steps) * 100
        elapsed = datetime.now() - self.start_time
        
        status_msg = f"{self.description}: {self.current_step}/{self.total_steps} ({percentage:.1f}%)"
        if message:
            status_msg += f" - {message}"
        
        if self.current_step > 0:
            eta_seconds = (elapsed.total_seconds() / self.current_step) * (self.total_steps - self.current_step)
            eta = format_duration(eta_seconds)
            status_msg += f" - ETA: {eta}"
        
        logging.info(status_msg)
    
    def complete(self):
        """Mark operation as complete."""
        duration = datetime.now() - self.start_time
        logging.info(f"{self.description} completed in {format_duration(duration.total_seconds())}")


class ExperimentTimer:
    """Context manager for timing experiments."""
    
    def __init__(self, name: str):
        """
        Initialize experiment timer.
        
        Args:
            name: Name of the experiment/operation
        """
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        """Enter context manager."""
        self.start_time = datetime.now()
        logging.info(f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        duration = datetime.now() - self.start_time
        if exc_type is None:
            logging.info(f"{self.name} completed successfully in {format_duration(duration.total_seconds())}")
        else:
            logging.error(f"{self.name} failed after {format_duration(duration.total_seconds())}")
            logging.error(f"Error: {exc_val}")


def verify_dependencies():
    """Verify that all required dependencies are installed."""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'xgboost', 'librosa', 
        'matplotlib', 'seaborn', 'tqdm', 'pyyaml', 'scikit-optimize'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(f"Missing required packages: {', '.join(missing_packages)}")
    
    logging.info("All required dependencies are available")
    return True