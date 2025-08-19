import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NoiseGenerator:
    """Generator for adding noise to audio signals."""
    
    def __init__(self, config: Dict):
        """
        Initialize noise generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.snr_levels = config['noise']['snr_levels']
        self.noise_type = config['noise']['noise_type']
        
        logger.info(f"Initialized noise generator with SNR levels: {self.snr_levels}")
        logger.info(f"Noise type: {self.noise_type}")
    
    def add_white_noise(self, clean_signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add white Gaussian noise to a clean signal at specified SNR.
        
        Args:
            clean_signal: Clean audio signal
            snr_db: Desired signal-to-noise ratio in dB
            
        Returns:
            Noisy signal
        """
        # Generate white noise
        noise = np.random.randn(len(clean_signal))
        
        # Calculate signal and noise power
        signal_power = np.mean(clean_signal**2)
        noise_power = np.mean(noise**2)
        
        # Handle edge cases
        if noise_power < 1e-10:
            noise_power = 1e-10
        if signal_power < 1e-10:
            logger.warning("Silent input signal detected")
            return noise * 0.001  # Return very quiet noise
        
        # Calculate scaling factor for desired SNR
        k = np.sqrt(signal_power / (noise_power * (10**(snr_db / 10))))
        scaled_noise = k * noise
        
        # Add noise to signal
        noisy_signal = clean_signal + scaled_noise
        
        return noisy_signal.astype(np.float32)
    
    def add_pink_noise(self, clean_signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add pink noise (1/f noise) to a clean signal.
        
        Args:
            clean_signal: Clean audio signal
            snr_db: Desired signal-to-noise ratio in dB
            
        Returns:
            Noisy signal
        """
        # Generate pink noise using spectral shaping
        white_noise = np.random.randn(len(clean_signal))
        
        # Apply 1/f shaping in frequency domain
        fft_white = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(len(white_noise))
        freqs[0] = 1e-10  # Avoid division by zero
        
        # Shape spectrum (1/f)
        pink_spectrum = fft_white / np.sqrt(np.abs(freqs))
        pink_noise = np.real(np.fft.ifft(pink_spectrum))
        
        # Normalize and scale to desired SNR
        signal_power = np.mean(clean_signal**2)
        noise_power = np.mean(pink_noise**2)
        
        if noise_power < 1e-10:
            noise_power = 1e-10
        if signal_power < 1e-10:
            return pink_noise * 0.001
        
        k = np.sqrt(signal_power / (noise_power * (10**(snr_db / 10))))
        scaled_noise = k * pink_noise
        
        return (clean_signal + scaled_noise).astype(np.float32)
    
    def add_noise(self, clean_signal: np.ndarray, snr_db: float, noise_type: str = None) -> np.ndarray:
        """
        Add noise to a clean signal.
        
        Args:
            clean_signal: Clean audio signal
            snr_db: Desired signal-to-noise ratio in dB
            noise_type: Type of noise ('white', 'pink'). If None, uses config default
            
        Returns:
            Noisy signal
        """
        if noise_type is None:
            noise_type = self.noise_type
        
        if noise_type == 'white':
            return self.add_white_noise(clean_signal, snr_db)
        elif noise_type == 'pink':
            return self.add_pink_noise(clean_signal, snr_db)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
    
    def calculate_actual_snr(self, clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
        """
        Calculate the actual SNR between clean and noisy signals.
        
        Args:
            clean_signal: Clean audio signal
            noisy_signal: Noisy audio signal
            
        Returns:
            Actual SNR in dB
        """
        noise = noisy_signal - clean_signal
        signal_power = np.mean(clean_signal**2)
        noise_power = np.mean(noise**2)
        
        if noise_power < 1e-10 or signal_power < 1e-10:
            return float('inf')
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db
    
    def process_dataset_with_noise(
        self,
        df: pd.DataFrame,
        feature_extractor,
        target_sr: int,
        save_path: str = None
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """
        Process entire dataset with noise at all configured SNR levels.
        
        Args:
            df: DataFrame containing audio file information
            feature_extractor: Fitted feature extractor
            target_sr: Target sampling rate
            save_path: Optional path to save processed data
            
        Returns:
            Dictionary mapping SNR levels to (features, labels) tuples
        """
        logger.info("Processing dataset with noise at multiple SNR levels...")
        
        noisy_datasets = {}
        
        for snr_db in self.snr_levels:
            logger.info(f"Processing SNR {snr_db} dB...")
            
            all_features = []
            all_labels = []
            
            for index, row in tqdm(df.iterrows(), total=len(df), desc=f"SNR {snr_db}dB"):
                try:
                    # Load clean audio
                    clean_signal, sr = librosa.load(row['full_path'], sr=target_sr, mono=True)
                    
                    # Add noise
                    noisy_signal = self.add_noise(clean_signal, snr_db)
                    
                    # Extract features
                    features, _ = feature_extractor.extract_features_from_signal(noisy_signal, target_sr)
                    
                    if not np.all(np.isnan(features)):
                        all_features.append(features)
                        all_labels.append(row['emotion_label'])
                    else:
                        logger.warning(f"Skipping {row['full_path']} for SNR {snr_db}dB (feature extraction failed)")
                        
                except Exception as e:
                    logger.error(f"Error processing {row['full_path']} at SNR {snr_db}dB: {e}")
                    continue
            
            if all_features:
                features_array = np.array(all_features)
                labels_array = np.array(all_labels)
                noisy_datasets[snr_db] = (features_array, labels_array)
                
                logger.info(f"SNR {snr_db}dB: {features_array.shape[0]} samples processed")
            else:
                logger.warning(f"No valid samples for SNR {snr_db}dB")
        
        return noisy_datasets
    
    def validate_snr_levels(self, test_signals: List[np.ndarray], num_samples: int = 10) -> Dict[float, float]:
        """
        Validate that noise addition achieves target SNR levels.
        
        Args:
            test_signals: List of clean test signals
            num_samples: Number of samples to test per SNR level
            
        Returns:
            Dictionary mapping target SNR to actual SNR statistics
        """
        logger.info("Validating SNR levels...")
        
        validation_results = {}
        
        for target_snr in self.snr_levels:
            actual_snrs = []
            
            # Test on subset of signals
            test_subset = test_signals[:min(num_samples, len(test_signals))]
            
            for clean_signal in test_subset:
                noisy_signal = self.add_noise(clean_signal, target_snr)
                actual_snr = self.calculate_actual_snr(clean_signal, noisy_signal)
                
                if not np.isinf(actual_snr):
                    actual_snrs.append(actual_snr)
            
            if actual_snrs:
                validation_results[target_snr] = {
                    'mean': np.mean(actual_snrs),
                    'std': np.std(actual_snrs),
                    'min': np.min(actual_snrs),
                    'max': np.max(actual_snrs)
                }
                
                logger.info(f"Target SNR {target_snr}dB: "
                           f"Actual {validation_results[target_snr]['mean']:.2f} ± "
                           f"{validation_results[target_snr]['std']:.2f}dB")
            else:
                logger.warning(f"No valid SNR calculations for target {target_snr}dB")
        
        return validation_results


class AudioCorruptor:
    """Extended audio corruption with various degradation types."""
    
    def __init__(self, config: Dict):
        """Initialize audio corruptor."""
        self.config = config
        self.noise_generator = NoiseGenerator(config)
    
    def add_reverberation(self, signal: np.ndarray, room_size: float = 0.5) -> np.ndarray:
        """
        Add simple reverberation effect.
        
        Args:
            signal: Input audio signal
            room_size: Room size parameter (0.0 to 1.0)
            
        Returns:
            Signal with reverberation
        """
        # Simple delay-based reverb
        delay_samples = int(0.05 * len(signal) * room_size)  # 50ms max delay
        decay_factor = 0.3 * room_size
        
        reverb_signal = signal.copy()
        if delay_samples > 0 and delay_samples < len(signal):
            delayed = np.zeros_like(signal)
            delayed[delay_samples:] = signal[:-delay_samples] * decay_factor
            reverb_signal = signal + delayed
        
        return reverb_signal
    
    def add_compression(self, signal: np.ndarray, ratio: float = 4.0, threshold: float = 0.5) -> np.ndarray:
        """
        Add dynamic range compression.
        
        Args:
            signal: Input audio signal
            ratio: Compression ratio
            threshold: Compression threshold
            
        Returns:
            Compressed signal
        """
        # Simple compression implementation
        abs_signal = np.abs(signal)
        mask = abs_signal > threshold
        
        compressed = signal.copy()
        compressed[mask] = np.sign(signal[mask]) * (
            threshold + (abs_signal[mask] - threshold) / ratio
        )
        
        return compressed
    
    def create_degradation_pipeline(self, degradations: List[str]) -> callable:
        """
        Create a pipeline of audio degradations.
        
        Args:
            degradations: List of degradation types
            
        Returns:
            Callable degradation function
        """
        def apply_degradations(signal: np.ndarray, **kwargs) -> np.ndarray:
            degraded = signal.copy()
            
            for degradation in degradations:
                if degradation == 'noise':
                    snr_db = kwargs.get('snr_db', 10)
                    degraded = self.noise_generator.add_noise(degraded, snr_db)
                elif degradation == 'reverb':
                    room_size = kwargs.get('room_size', 0.5)
                    degraded = self.add_reverberation(degraded, room_size)
                elif degradation == 'compression':
                    ratio = kwargs.get('ratio', 4.0)
                    threshold = kwargs.get('threshold', 0.5)
                    degraded = self.add_compression(degraded, ratio, threshold)
                else:
                    logger.warning(f"Unknown degradation type: {degradation}")
            
            return degraded
        
        return apply_degradations