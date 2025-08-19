"""
Feature extraction module for speech emotion recognition.

This module provides comprehensive audio feature extraction including
F0, energy, MFCCs, spectral features, and their temporal derivatives.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import librosa
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Comprehensive audio feature extractor for SER."""
    
    def __init__(self, config: Dict):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.audio_config = config['audio']
        self.features_config = config['features']
        
        # Audio processing parameters
        self.target_sr = config['data']['target_sr']
        self.n_fft = self.audio_config['n_fft']
        self.hop_length = self.audio_config['hop_length']
        self.n_mfcc = self.audio_config['n_mfcc']
        
        logger.info(f"Initialized feature extractor with SR={self.target_sr}, "
                   f"n_fft={self.n_fft}, hop_length={self.hop_length}")
    
    def extract_f0_features(self, audio_signal: np.ndarray, sr: int) -> Tuple[List[float], List[str]]:
        """
        Extract fundamental frequency (F0) features.
        
        Args:
            audio_signal: Audio signal
            sr: Sample rate
            
        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        feature_names = ['meanF0', 'stdF0', 'medianF0', 'minF0', 'maxF0']
        
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio_signal,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )
            
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 0:
                features.extend([
                    np.mean(f0_voiced),
                    np.std(f0_voiced),
                    np.median(f0_voiced),
                    np.min(f0_voiced),
                    np.max(f0_voiced)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
                
        except Exception as e:
            logger.warning(f"F0 extraction failed: {e}")
            features.extend([0, 0, 0, 0, 0])
        
        return features, feature_names
    
    def extract_energy_features(self, audio_signal: np.ndarray) -> Tuple[List[float], List[str]]:
        """
        Extract RMS energy features.
        
        Args:
            audio_signal: Audio signal
            
        Returns:
            Tuple of (features, feature_names)
        """
        feature_names = ['meanEnergy', 'stdEnergy', 'medianEnergy', 'minEnergy', 'maxEnergy']
        
        try:
            rms_energy = librosa.feature.rms(
                y=audio_signal,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            features = [
                np.mean(rms_energy),
                np.std(rms_energy),
                np.median(rms_energy),
                np.min(rms_energy),
                np.max(rms_energy)
            ]
            
        except Exception as e:
            logger.warning(f"Energy extraction failed: {e}")
            features = [0, 0, 0, 0, 0]
        
        return features, feature_names
    
    def extract_mfcc_features(self, audio_signal: np.ndarray, sr: int) -> Tuple[List[float], List[str]]:
        """
        Extract MFCC features including deltas and delta-deltas.
        
        Args:
            audio_signal: Audio signal
            sr: Sample rate
            
        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        feature_names = []
        
        try:
            # Base MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_signal,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Calculate means and stds for base MFCCs
            means = np.mean(mfccs, axis=1)
            stds = np.std(mfccs, axis=1)
            features.extend(means)
            features.extend(stds)
            
            # Add feature names for base MFCCs
            for j in range(self.n_mfcc):
                feature_names.append(f'mfcc{j}_mean')
                feature_names.append(f'mfcc{j}_std')
            
            if self.features_config['include_delta_mfcc']:
                # Delta MFCCs
                delta_mfccs = librosa.feature.delta(mfccs)
                delta_means = np.mean(delta_mfccs, axis=1)
                delta_stds = np.std(delta_mfccs, axis=1)
                features.extend(delta_means)
                features.extend(delta_stds)
                
                # Add feature names for delta MFCCs
                for j in range(self.n_mfcc):
                    feature_names.append(f'delta_mfcc{j}_mean')
                    feature_names.append(f'delta_mfcc{j}_std')
            
            if self.features_config['include_delta2_mfcc']:
                # Delta-Delta MFCCs
                delta2_mfccs = librosa.feature.delta(mfccs, order=2)
                delta2_means = np.mean(delta2_mfccs, axis=1)
                delta2_stds = np.std(delta2_mfccs, axis=1)
                features.extend(delta2_means)
                features.extend(delta2_stds)
                
                # Add feature names for delta-delta MFCCs
                for j in range(self.n_mfcc):
                    feature_names.append(f'delta2_mfcc{j}_mean')
                    feature_names.append(f'delta2_mfcc{j}_std')
            
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            # Return zeros for expected number of features
            expected_features = self.n_mfcc * 2  # base means + stds
            if self.features_config['include_delta_mfcc']:
                expected_features += self.n_mfcc * 2
            if self.features_config['include_delta2_mfcc']:
                expected_features += self.n_mfcc * 2
            features = [0] * expected_features
            # Generate corresponding feature names
            feature_names = [f'mfcc_error_{i}' for i in range(expected_features)]
        
        return features, feature_names
    
    def extract_spectral_features(self, audio_signal: np.ndarray, sr: int) -> Tuple[List[float], List[str]]:
        """
        Extract spectral features.
        
        Args:
            audio_signal: Audio signal
            sr: Sample rate
            
        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        feature_names = []
        
        # Zero Crossing Rate
        if self.features_config['include_zcr']:
            try:
                zcr = librosa.feature.zero_crossing_rate(
                    y=audio_signal,
                    frame_length=self.n_fft,
                    hop_length=self.hop_length
                )[0]
                features.extend([np.mean(zcr), np.std(zcr)])
                feature_names.extend(['meanZCR', 'stdZCR'])
            except Exception as e:
                logger.warning(f"ZCR extraction failed: {e}")
                features.extend([0, 0])
                feature_names.extend(['meanZCR', 'stdZCR'])
        
        # Spectral Centroid
        if self.features_config['include_spectral_centroid']:
            try:
                spec_cent = librosa.feature.spectral_centroid(
                    y=audio_signal,
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )[0]
                features.extend([np.mean(spec_cent), np.std(spec_cent)])
                feature_names.extend(['meanSpecCent', 'stdSpecCent'])
            except Exception as e:
                logger.warning(f"Spectral centroid extraction failed: {e}")
                features.extend([0, 0])
                feature_names.extend(['meanSpecCent', 'stdSpecCent'])
        
        return features, feature_names
    
    def extract_features_from_signal(self, audio_signal: np.ndarray, sr: int) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all configured features from an audio signal.
        
        Args:
            audio_signal: Audio signal
            sr: Sample rate
            
        Returns:
            Tuple of (feature_array, feature_names)
        """
        all_features = []
        all_feature_names = []
        
        try:
            # F0 Features
            if self.features_config['include_f0']:
                f0_features, f0_names = self.extract_f0_features(audio_signal, sr)
                all_features.extend(f0_features)
                all_feature_names.extend(f0_names)
            
            # Energy Features
            if self.features_config['include_energy']:
                energy_features, energy_names = self.extract_energy_features(audio_signal)
                all_features.extend(energy_features)
                all_feature_names.extend(energy_names)
            
            # MFCC Features
            if self.features_config['include_mfcc']:
                mfcc_features, mfcc_names = self.extract_mfcc_features(audio_signal, sr)
                all_features.extend(mfcc_features)
                all_feature_names.extend(mfcc_names)
            
            # Spectral Features
            spectral_features, spectral_names = self.extract_spectral_features(audio_signal, sr)
            all_features.extend(spectral_features)
            all_feature_names.extend(spectral_names)
            
            return np.array(all_features), all_feature_names
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return NaN array of expected size
            expected_size = self._calculate_expected_feature_size()
            return np.full(expected_size, np.nan), []
    
    def _calculate_expected_feature_size(self) -> int:
        """Calculate expected feature vector size based on configuration."""
        size = 0
        
        if self.features_config['include_f0']:
            size += 5  # F0 features
        if self.features_config['include_energy']:
            size += 5  # Energy features
        if self.features_config['include_mfcc']:
            size += self.n_mfcc * 2  # Base MFCCs (mean + std)
            if self.features_config['include_delta_mfcc']:
                size += self.n_mfcc * 2  # Delta MFCCs
            if self.features_config['include_delta2_mfcc']:
                size += self.n_mfcc * 2  # Delta-delta MFCCs
        if self.features_config['include_zcr']:
            size += 2  # ZCR features
        if self.features_config['include_spectral_centroid']:
            size += 2  # Spectral centroid features
        
        return size
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        label_encoder: Optional[LabelEncoder] = None,
        fit_label_encoder: bool = False,
        feature_means: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, Optional[LabelEncoder], List[str], Optional[pd.Series]]:
        """
        Process a DataFrame of audio files and extract features.
        
        Args:
            df: DataFrame containing audio file paths and labels
            label_encoder: Pre-fitted label encoder (optional)
            fit_label_encoder: Whether to fit a new label encoder
            feature_means: Feature means for imputation (from training data)
            feature_names: Expected feature names
            
        Returns:
            Tuple of (features_df, labels_encoded, label_encoder, feature_names, calculated_means)
        """
        logger.info(f"Extracting features for {len(df)} files...")
        
        all_features_list = []
        labels_list = []
        current_feature_names = feature_names
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            try:
                # Load audio
                audio_signal, sr = librosa.load(
                    row['full_path'],
                    sr=self.target_sr,
                    mono=True
                )
                
                # Extract features
                features_array, extracted_names = self.extract_features_from_signal(audio_signal, sr)
                
                if not np.all(np.isnan(features_array)):
                    all_features_list.append(features_array)
                    labels_list.append(row['emotion_label'])
                    
                    # Set feature names from first valid extraction
                    if current_feature_names is None and extracted_names:
                        current_feature_names = extracted_names
                else:
                    logger.warning(f"Skipping file due to extraction error: {row['full_path']}")
                    
            except Exception as e:
                logger.error(f"Error processing {row['full_path']}: {e}")
                continue
        
        if not all_features_list:
            logger.error("No features extracted!")
            return pd.DataFrame(), np.array([]), label_encoder, [], None
        
        # Create features DataFrame
        features_df = pd.DataFrame(all_features_list, columns=current_feature_names)
        
        # Handle missing values
        calculated_means = None
        if feature_means is not None:
            # Use provided means (from training data)
            features_df = features_df.fillna(feature_means)
            logger.info("NaNs imputed using provided feature means")
        else:
            # Calculate means from current data (should be training set)
            calculated_means = features_df.mean()
            features_df = features_df.fillna(calculated_means)
            logger.info("NaNs imputed using current data's feature means")
        
        # Handle labels
        if fit_label_encoder:
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(labels_list)
        else:
            if label_encoder is None:
                raise ValueError("Label encoder must be provided if fit_label_encoder is False")
            labels_encoded = label_encoder.transform(labels_list)
        
        logger.info(f"Features shape: {features_df.shape}, Labels shape: {labels_encoded.shape}")
        
        return features_df, labels_encoded, label_encoder, current_feature_names, calculated_means