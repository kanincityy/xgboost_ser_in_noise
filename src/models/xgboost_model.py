"""
XGBoost model implementation for speech emotion recognition.

This module provides a complete XGBoost-based classifier with
hyperparameter optimization and comprehensive evaluation capabilities.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

logger = logging.getLogger(__name__)


class XGBoostSERModel:
    """XGBoost-based Speech Emotion Recognition model."""
    
    def __init__(self, config: Dict):
        """
        Initialize XGBoost SER model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config['model']['xgboost']
        self.hyperopt_config = config['hyperopt']
        
        # Model components
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: Optional[List[str]] = None
        self.feature_means: Optional[pd.Series] = None
        
        # Training metadata
        self.is_fitted = False
        self.best_params: Optional[Dict] = None
        self.cv_results: Optional[Dict] = None
        
        logger.info("Initialized XGBoost SER model")
    
    def _create_base_model(self) -> xgb.XGBClassifier:
        """Create base XGBoost model with default parameters."""
        return xgb.XGBClassifier(**self.model_config)
    
    def _get_hyperparameter_space(self) -> Dict:
        """Define hyperparameter search space."""
        return {
            'learning_rate': Real(0.001, 0.1, prior='log-uniform'),
            'max_depth': Integer(2, 6),
            'n_estimators': Integer(100, 2000),
            'subsample': Real(0.5, 1.0, prior='uniform'),
            'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
            'colsample_bylevel': Real(0.3, 1.0, prior='uniform'),
            'max_delta_step': Integer(0, 10),
            'gamma': Real(0.0, 2.0, prior='uniform'),
            'min_child_weight': Integer(1, 10),
            'reg_alpha': Real(1e-5, 1.0, prior='log-uniform'),
            'reg_lambda': Real(1e-6, 5.0, prior='log-uniform')
        }
    
    def prepare_training_data(
        self,
        train_features_df: pd.DataFrame,
        train_labels: np.ndarray,
        val_features_df: pd.DataFrame,
        val_labels: np.ndarray,
        feature_names: List[str],
        fit_preprocessing: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and preprocess training data.
        
        Args:
            train_features_df: Training features DataFrame
            train_labels: Training labels
            val_features_df: Validation features DataFrame  
            val_labels: Validation labels
            feature_names: List of feature names
            fit_preprocessing: Whether to fit preprocessing components
            
        Returns:
            Tuple of (combined_features, combined_labels)
        """
        logger.info("Preparing training data...")
        
        # Store feature names and preprocessing components
        self.feature_names = feature_names
        
        if fit_preprocessing:
            # Fit label encoder
            all_labels = np.concatenate([train_labels, val_labels])
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_labels)
            
            # Fit scaler on training data only
            self.scaler = StandardScaler()
            self.scaler.fit(train_features_df)
            
            # Store feature means for later use with noisy data
            self.feature_means = train_features_df.mean()
        
        # Transform data
        train_features_scaled = self.scaler.transform(train_features_df)
        val_features_scaled = self.scaler.transform(val_features_df)
        train_labels_encoded = self.label_encoder.transform(train_labels)
        val_labels_encoded = self.label_encoder.transform(val_labels)
        
        # Combine training and validation data for hyperparameter tuning
        combined_features = np.vstack([train_features_scaled, val_features_scaled])
        combined_labels = np.concatenate([train_labels_encoded, val_labels_encoded])
        
        logger.info(f"Prepared {len(combined_features)} samples for training")
        return combined_features, combined_labels
    
    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        cv_strategy: Optional[Any] = None
    ) -> Dict:
        """
        Optimize hyperparameters using Bayesian search.
        
        Args:
            X: Feature matrix
            y: Target labels
            groups: Group labels for cross-validation
            cv_strategy: Cross-validation strategy
            
        Returns:
            Best hyperparameters
        """
        logger.info("Starting hyperparameter optimization...")
        
        # Set up cross-validation strategy
        if cv_strategy is None:
            # Create gender-based stratification for CV
            speaker_genders = np.array(['female' if g % 2 == 0 else 'male' for g in groups])
            cv_strategy = StratifiedGroupKFold(
                n_splits=self.hyperopt_config['cv_folds'],
                shuffle=True,
                random_state=self.hyperopt_config['random_state']
            )
            cv_iterator = list(cv_strategy.split(X, speaker_genders, groups))
        else:
            cv_iterator = cv_strategy
        
        # Create base model
        base_model = self._create_base_model()
        base_model.num_class = len(self.label_encoder.classes_)
        
        # Set up Bayesian optimization
        param_space = self._get_hyperparameter_space()
        
        from sklearn.metrics import make_scorer, log_loss
        scorer = make_scorer(
            log_loss,
            greater_is_better=False,
            response_method="predict_proba",
            labels=np.arange(len(self.label_encoder.classes_))
        )
        
        opt = BayesSearchCV(
            estimator=base_model,
            search_spaces=param_space,
            n_iter=self.hyperopt_config['n_iter'],
            cv=cv_iterator,
            scoring=scorer,
            optimizer_kwargs={'base_estimator': 'GP'},
            random_state=self.hyperopt_config['random_state'],
            verbose=2,
            n_jobs=-1
        )
        
        # Fit optimization
        opt.fit(X, y)
        
        # Store results
        self.best_params = dict(opt.best_params_)
        self.cv_results = opt.cv_results_
        self.model = opt.best_estimator_
        self.is_fitted = True
        
        logger.info(f"Best cross-validation score: {opt.best_score_:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def train(
        self,
        train_features_df: pd.DataFrame,
        train_labels: np.ndarray,
        val_features_df: pd.DataFrame,
        val_labels: np.ndarray,
        feature_names: List[str],
        groups: np.ndarray,
        optimize_hyperparams: bool = True
    ) -> 'XGBoostSERModel':
        """
        Train the XGBoost model.
        
        Args:
            train_features_df: Training features
            train_labels: Training labels
            val_features_df: Validation features
            val_labels: Validation labels
            feature_names: Feature names
            groups: Group identifiers for CV
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Self for method chaining
        """
        logger.info("Training XGBoost model...")
        
        # Prepare data
        X, y = self.prepare_training_data(
            train_features_df, train_labels,
            val_features_df, val_labels,
            feature_names, fit_preprocessing=True
        )
        
        if optimize_hyperparams:
            # Optimize hyperparameters
            self.optimize_hyperparameters(X, y, groups)
        else:
            # Train with default parameters
            self.model = self._create_base_model()
            self.model.num_class = len(self.label_encoder.classes_)
            self.model.fit(X, y)
            self.is_fitted = True
        
        logger.info("Model training completed")
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Make probability predictions on features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(features)
    
    def evaluate(self, features: np.ndarray, true_labels: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            features: Feature matrix
            true_labels: True labels
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(features)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Classification report
        report = classification_report(
            true_labels, predictions,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(
            true_labels, predictions,
            labels=np.arange(len(self.label_encoder.classes_))
        )
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions
        }
    
    def process_noisy_data(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process noisy data using fitted preprocessing.
        
        Args:
            features_df: Features DataFrame
            labels: Labels array
            
        Returns:
            Tuple of (processed_features, encoded_labels)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before processing data")
        
        # Impute missing values using training means
        features_df_imputed = features_df.fillna(self.feature_means)
        
        # Scale features
        features_scaled = self.scaler.transform(features_df_imputed)
        
        # Encode labels
        labels_encoded = self.label_encoder.transform(labels)
        
        return features_scaled, labels_encoded
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, save_dir: str) -> None:
        """
        Save trained model and preprocessing components.
        
        Args:
            save_dir: Directory to save model components
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Save model
        model_path = save_path / 'xgboost_model.joblib'
        joblib.dump(self.model, model_path)
        
        # Save preprocessing components
        scaler_path = save_path / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        encoder_path = save_path / 'label_encoder.joblib'
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save feature names and means
        features_path = save_path / 'feature_names.joblib'
        joblib.dump(self.feature_names, features_path)
        
        means_path = save_path / 'feature_means.pkl'
        self.feature_means.to_pickle(means_path)
        
        # Save metadata
        metadata = {
            'best_params': self.best_params,
            'is_fitted': self.is_fitted,
            'num_classes': len(self.label_encoder.classes_),
            'class_names': list(self.label_encoder.classes_)
        }
        
        metadata_path = save_path / 'model_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {save_dir}")
    
    def load_model(self, save_dir: str) -> 'XGBoostSERModel':
        """
        Load trained model and preprocessing components.
        
        Args:
            save_dir: Directory containing saved model
            
        Returns:
            Self for method chaining
        """
        save_path = Path(save_dir)
        
        # Load model
        model_path = save_path / 'xgboost_model.joblib'
        self.model = joblib.load(model_path)
        
        # Load preprocessing components
        scaler_path = save_path / 'scaler.joblib'
        self.scaler = joblib.load(scaler_path)
        
        encoder_path = save_path / 'label_encoder.joblib'
        self.label_encoder = joblib.load(encoder_path)
        
        # Load feature names and means
        features_path = save_path / 'feature_names.joblib'
        self.feature_names = joblib.load(features_path)
        
        means_path = save_path / 'feature_means.pkl'
        self.feature_means = pd.read_pickle(means_path)
        
        # Load metadata
        metadata_path = save_path / 'model_metadata.pkl'
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.best_params = metadata.get('best_params')
        self.is_fitted = metadata.get('is_fitted', True)
        
        logger.info(f"Model loaded from {save_dir}")
        return self