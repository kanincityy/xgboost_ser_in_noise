import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tempfile
import shutil

from models.xgboost_model import XGBoostSERModel


class TestXGBoostSERModel:
    """Test cases for XGBoostSERModel."""
    
    @pytest.fixture
    def config(self):
        """Sample configuration."""
        return {
            'model': {
                'xgboost': {
                    'objective': 'multi:softmax',
                    'num_class': 4,
                    'tree_method': 'hist',
                    'device': 'cpu',
                    'random_state': 42,
                    'n_estimators': 10  # Small for testing
                }
            },
            'hyperopt': {
                'n_iter': 2,  # Small for testing
                'cv_folds': 2,
                'random_state': 42
            }
        }
    
    @pytest.fixture
    def model(self, config):
        """Create model instance."""
        return XGBoostSERModel(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        n_samples, n_features = 100, 20
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice(['angry', 'happy', 'sad', 'calm'], size=n_samples)
        
        # Create DataFrames
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        return X_df, y, feature_names
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert not model.is_fitted
        assert model.model is None
        assert model.scaler is None
        assert model.label_encoder is None
    
    def test_base_model_creation(self, model):
        """Test base model creation."""
        base_model = model._create_base_model()
        assert base_model is not None
        assert base_model.get_params()['random_state'] == 42
    
    def test_hyperparameter_space(self, model):
        """Test hyperparameter space definition."""
        param_space = model._get_hyperparameter_space()
        assert isinstance(param_space, dict)
        assert 'learning_rate' in param_space
        assert 'max_depth' in param_space
        assert 'n_estimators' in param_space
    
    def test_data_preparation(self, model, sample_data):
        """Test data preparation."""
        X_train, y_train, feature_names = sample_data
        X_val, y_val = X_train[:20], y_train[:20]  # Use subset for validation
        
        X_combined, y_combined = model.prepare_training_data(
            X_train, y_train, X_val, y_val, feature_names, fit_preprocessing=True
        )
        
        assert model.scaler is not None
        assert model.label_encoder is not None
        assert model.feature_names == feature_names
        assert len(X_combined) == len(X_train) + len(X_val)
    
    def test_training_without_optimization(self, model, sample_data):
        """Test training without hyperparameter optimization."""
        X_df, y, feature_names = sample_data
        
        # Split data
        split_idx = len(X_df) // 2
        X_train, X_val = X_df[:split_idx], X_df[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create groups
        groups = np.arange(len(X_df))
        
        model.train(
            X_train, y_train, X_val, y_val,
            feature_names, groups, optimize_hyperparams=False
        )
        
        assert model.is_fitted
        assert model.model is not None
    
    def test_prediction(self, model, sample_data):
        """Test model prediction."""
        X_df, y, feature_names = sample_data
        
        # Quick training
        split_idx = len(X_df) // 2
        X_train, X_val = X_df[:split_idx], X_df[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        groups = np.arange(len(X_df))
        
        model.train(
            X_train, y_train, X_val, y_val,
            feature_names, groups, optimize_hyperparams=False
        )
        
        # Test prediction
        X_test_scaled, y_test_encoded = model.process_noisy_data(X_val, y_val)
        predictions = model.predict(X_test_scaled)
        
        assert len(predictions) == len(X_val)
        assert all(pred in range(len(np.unique(y))) for pred in predictions)
    
    def test_evaluation(self, model, sample_data):
        """Test model evaluation."""
        X_df, y, feature_names = sample_data
        
        # Quick training
        split_idx = len(X_df) // 2
        X_train, X_val = X_df[:split_idx], X_df[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        groups = np.arange(len(X_df))
        
        model.train(
            X_train, y_train, X_val, y_val,
            feature_names, groups, optimize_hyperparams=False
        )
        
        # Test evaluation
        X_test_scaled, y_test_encoded = model.process_noisy_data(X_val, y_val)
        results = model.evaluate(X_test_scaled, y_test_encoded)
        
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'confusion_matrix' in results
        assert 0 <= results['accuracy'] <= 1
    
    def test_save_and_load_model(self, model, sample_data):
        """Test model saving and loading."""
        X_df, y, feature_names = sample_data
        
        # Quick training
        split_idx = len(X_df) // 2
        X_train, X_val = X_df[:split_idx], X_df[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        groups = np.arange(len(X_df))
        
        model.train(
            X_train, y_train, X_val, y_val,
            feature_names, groups, optimize_hyperparams=False
        )
        
        # Test save and load
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_model(temp_dir)
            
            # Create new model instance and load
            new_model = XGBoostSERModel(model.config)
            new_model.load_model(temp_dir)
            
            assert new_model.is_fitted
            assert new_model.feature_names == model.feature_names
            assert len(new_model.label_encoder.classes_) == len(model.label_encoder.classes_)
    
    def test_feature_importance(self, model, sample_data):
        """Test feature importance extraction."""
        X_df, y, feature_names = sample_data
        
        # Quick training
        split_idx = len(X_df) // 2
        X_train, X_val = X_df[:split_idx], X_df[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        groups = np.arange(len(X_df))
        
        model.train(
            X_train, y_train, X_val, y_val,
            feature_names, groups, optimize_hyperparams=False
        )
        
        importance_df = model.get_feature_importance(top_n=5)
        
        assert len(importance_df) == 5
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert all(imp >= 0 for imp in importance_df['importance'])