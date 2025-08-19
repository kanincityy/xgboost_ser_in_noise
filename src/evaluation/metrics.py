"""
Comprehensive evaluation and metrics module for speech emotion recognition.

This module provides statistical analysis, confidence interval calculations,
and performance degradation analysis for SER models under various conditions.
"""

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
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
        }
        logger.info(f"Calculated basic metrics: {metrics}")
        return metrics