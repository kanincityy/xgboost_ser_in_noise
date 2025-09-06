import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import StratifiedGroupKFold

from config import (
    DATA_PATH, OUTPUT_PATH, GLOBAL_FEATURES_CLEAN_DIR,
    N_ITER_BAYES, CV_SPLITS_BAYES, XGB_FIXED_PARAMS
)

def main():
    """Trains the XGBoost model using Bayesian hyperparameter search."""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 1. Load data
    X_train = np.load(os.path.join(GLOBAL_FEATURES_CLEAN_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(GLOBAL_FEATURES_CLEAN_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(GLOBAL_FEATURES_CLEAN_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(GLOBAL_FEATURES_CLEAN_DIR, 'y_val.npy'))

    # Combine train and validation sets into a single pool for cross-validation
    X_pool = np.concatenate((X_train, X_val), axis=0)
    y_pool = np.concatenate((y_train, y_val), axis=0)
    
    # Load speaker groups for cross-validation
    train_df = pd.read_pickle(os.path.join(DATA_PATH, 'train_df.pkl'))
    val_df = pd.read_pickle(os.path.join(DATA_PATH, 'val_df.pkl'))
    pool_groups = pd.concat([train_df['speaker_id'], val_df['speaker_id']], ignore_index=True).values
    
    # For stratification, create a gender array
    speaker_genders = pd.Series(pool_groups).apply(lambda x: 'female' if x % 2 == 0 else 'male').values
    
    # 2. Define search space and CV strategy
    param_space = {
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'max_depth': Integer(3, 8),
        'n_estimators': Integer(200, 1500),
        'subsample': Real(0.6, 1.0, prior='uniform'),
        'colsample_bytree': Real(0.6, 1.0, prior='uniform'),
        'gamma': Real(0.0, 1.0, prior='uniform'),
        'reg_alpha': Real(1e-5, 1.0, prior='log-uniform'),
        'reg_lambda': Real(1e-5, 1.0, prior='log-uniform')
    }
    
    cv_strategy = StratifiedGroupKFold(n_splits=CV_SPLITS_BAYES, shuffle=True, random_state=42)
    
    # 3. Initialize XGBoost model and BayesSearchCV
    le = joblib.load(os.path.join(DATA_PATH, "emotion_label_encoder.joblib"))
    xgb_model = xgb.XGBClassifier(num_class=len(le.classes_), **XGB_FIXED_PARAMS)
    
    opt = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=param_space,
        n_iter=N_ITER_BAYES,
        cv=cv_strategy,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # 4. Run the optimization
    print("Starting Bayesian hyperparameter optimization...")
    opt.fit(X_pool, y_pool, groups=pool_groups)
    
    # 5. Save the best model
    best_model = opt.best_estimator_
    model_save_path = os.path.join(OUTPUT_PATH, 'best_xgb_model.joblib')
    joblib.dump(best_model, model_save_path)
    
    print("\nOptimization finished.")
    print(f"Best cross-validation accuracy: {opt.best_score_:.4f}")
    print("Best Parameters:", opt.best_params_)
    print(f"Best model saved to: {model_save_path}")

if __name__ == '__main__':
    main()