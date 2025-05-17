import polars as pl
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import joblib

def load_processed_data(file_path):
    """Load preprocessed data"""
    df = pl.read_csv(file_path)
    return df

def perform_cross_validation(X, y, model, n_splits=5):
    """Perform cross-validation"""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for i, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        cv_scores.append(score)
        print(f"Fold {i+1}: AUC = {score:.4f}")
    
    print(f"Mean AUC: {np.mean(cv_scores):.4f}, Std: {np.std(cv_scores):.4f}")
    return cv_scores

def hyperparameter_tuning(X, y):
    """Perform hyperparameter tuning with GridSearchCV"""
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'num_leaves': [31, 50, 100],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        random_state=42
    )
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best AUC score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def main():
    # Load preprocessed data
    df = load_processed_data('preprocessed_data.csv')
    
    # Convert to pandas for sklearn compatibility
    df_pd = df.to_pandas()
    
    # Split features and target
    X = df_pd.drop(columns=['TARGET'])
    y = df_pd['TARGET']
    
    # Perform cross-validation with basic model
    base_model = lgb.LGBMClassifier(random_state=42)
    print("Performing cross-validation with base model...")
    cv_scores = perform_cross_validation(X, y, base_model)
    
    # Perform hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    best_model, best_params = hyperparameter_tuning(X, y)
    
    # Save best model
    joblib.dump(best_model, 'best_credit_risk_model.pkl')
    print("Best model saved to best_credit_risk_model.pkl")

if __name__ == "__main__":
    main()