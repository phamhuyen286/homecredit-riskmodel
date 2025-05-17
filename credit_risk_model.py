import polars as pl
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Try to import SMOTE, but continue if not available
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("Warning: imblearn not installed. SMOTE will not be available.")
    print("To install: pip install imbalanced-learn")
    SMOTE_AVAILABLE = False

# Load data
def load_data(file_path):
    # Polars is faster than pandas for large datasets
    df = pl.read_csv(file_path)
    print(f"Data loaded with shape: {df.shape}")
    return df

# Basic exploration
def explore_data(df):
    # Display basic info
    print("Data overview:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    
    # Check for missing values
    missing = df.null_count()
    print("\nMissing values per column:")
    # Fix the filter expression to properly handle multiple columns
    missing_cols = missing.select([
        pl.col(col_name) for col_name in missing.columns 
        if missing[col_name][0] > 0
    ])
    if missing_cols.width > 0:
        print(missing_cols)
    else:
        print("No missing values found")
    
    # Target distribution
    if 'TARGET' in df.columns:
        target_counts = df.group_by('TARGET').agg(pl.count())
        print("\nTarget distribution:")
        print(target_counts)
    
    return df

def preprocess_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the data by handling missing values, outliers, and encoding."""
    df_processed = df.clone() # Work on a clone

    # Impute numerical columns with median and handle outliers
    numerical_cols = [col for col, dtype in df_processed.schema.items() if dtype.is_numeric() and col != 'TARGET']
    
    for col in numerical_cols:
        # Impute with median
        col_median = df_processed.select(pl.col(col).median()).item()
        
        if col_median is None:
            fill_value_for_num = 0.0
            df_processed = df_processed.with_columns(pl.col(col).fill_null(fill_value_for_num))
        else:
            df_processed = df_processed.with_columns(pl.col(col).fill_null(col_median))
            
        # Cap outliers at 99th percentile (only for specified columns)
        if col in ["actualdpd_943P", "childnum_21L", "credamount_590A", "mainoccupationinc_437A"]:
            if df_processed[col].null_count() < len(df_processed): # Check if column is not all nulls
                percentile_99_series = df_processed.select(pl.col(col).quantile(0.99, "midpoint")).get_column(col)
                if not percentile_99_series.is_empty() and percentile_99_series[0] is not None:
                    percentile_99 = percentile_99_series[0]
                    
                    # Ensure upper_bound is not less than lower_bound.
                    actual_upper_bound = max(0.0, percentile_99) 
                    
                    df_processed = df_processed.with_columns(
                        pl.col(col).clip(lower_bound=0.0, upper_bound=actual_upper_bound)
                    )

        # Log-transform for skewed columns
        if col in ["credamount_590A", "mainoccupationinc_437A", "annuity_853A"]:
            df_processed = df_processed.with_columns(
                (pl.when(pl.col(col) >= 0).then(pl.col(col)).otherwise(0) + 1).log().alias(f"log_{col}")
            )

    # Impute existing categorical columns with mode
    categorical_cols = [col for col, dtype in df_processed.schema.items() 
                       if dtype == pl.Categorical or dtype == pl.Utf8 and col != 'TARGET']
    for col in categorical_cols:
        try:
            mode_series = df_processed.select(pl.col(col).mode().first()).get_column(col)
            
            if mode_series.is_empty() or mode_series[0] is None:
                actual_mode_to_fill = "Unknown"
            else:
                actual_mode_to_fill = mode_series[0]
                
            df_processed = df_processed.with_columns(pl.col(col).fill_null(actual_mode_to_fill))
        except Exception as e:
            print(f"Warning: Could not process column {col}: {e}")

    # Handle string columns (cast to categorical after imputation)
    string_cols = ["education_1138M", "familystate_726L", "credtype_587L", "profession_152M", 
                   "cancelreason_3545846M", "district_544M", "postype_4733339M", 
                   "rejectreason_755M", "rejectreasonclient_4145042M", "status_219L"]
    for col in string_cols:
        if col in df_processed.columns:
            try:
                if df_processed.schema[col] == pl.Categorical:
                    continue

                mode_series = df_processed.select(pl.col(col).mode().first()).get_column(col)

                if mode_series.is_empty() or mode_series[0] is None:
                    actual_mode_to_fill = "Unknown"
                else:
                    actual_mode_to_fill = mode_series[0]
                
                df_processed = df_processed.with_columns(
                    pl.col(col).fill_null(actual_mode_to_fill).cast(pl.Categorical)
                )
            except Exception as e:
                print(f"Warning: Could not process string column {col}: {e}")

    # Convert time columns
    time_cols = ["employedfrom_700D", "dtlastpmt_581D", "creationdate_885D", "dtlastpmtallstes_3545839D", 
                 "firstnonzeroinstldate_307D", "dateactivated_425D", "approvaldate_319D"]
    for col in time_cols:
        if col in df_processed.columns:
            try:
                if df_processed.schema[col] == pl.Utf8: # If it's a string that should be a number
                     df_processed = df_processed.with_columns(
                        pl.col(col).cast(pl.Float64, strict=False) # Try to cast to float, nullify if fails
                     )

                # Now assume it's a numeric type (or became one)
                if df_processed.schema[col].is_numeric():
                    df_processed = df_processed.with_columns(
                        (pl.col(col).fill_null(0) / -365.25).alias(f"{col}_years")
                    )
            except Exception as e:
                print(f"Warning: Could not process time column {col}: {e}")

    # Feature engineering
    # Create ratio features if columns exist
    if all(col in df_processed.columns for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT']):
        df_processed = df_processed.with_columns(
            (pl.col('AMT_CREDIT') / pl.col('AMT_INCOME_TOTAL')).alias('CREDIT_INCOME_RATIO')
        )
    
    # Create age feature if DAYS_BIRTH exists
    if 'DAYS_BIRTH' in df_processed.columns:
        df_processed = df_processed.with_columns(
            (pl.col('DAYS_BIRTH') / -365.25).alias('AGE_YEARS')
        )

    # Clean up memory
    gc.collect()
    
    return df_processed

def feature_engineering(df):
    # Additional feature engineering beyond what's in preprocess_data
    
    # Create interaction features
    numeric_cols = [col for col, dtype in df.schema.items() 
                   if dtype.is_numeric() and col != 'TARGET' and not col.startswith('log_')]
    
    # Select a subset of important numeric columns for interactions to avoid explosion
    important_numeric_cols = numeric_cols[:5]  # Just use first 5 for example
    
    # Create some interaction terms
    for i in range(len(important_numeric_cols)):
        for j in range(i+1, len(important_numeric_cols)):
            col1 = important_numeric_cols[i]
            col2 = important_numeric_cols[j]
            try:
                # Multiplication interaction
                df = df.with_columns(
                    (pl.col(col1) * pl.col(col2)).alias(f"{col1}_mul_{col2}")
                )
                
                # Division interaction (with safeguard against division by zero)
                df = df.with_columns(
                    (pl.col(col1) / (pl.col(col2) + 0.001)).alias(f"{col1}_div_{col2}")
                )
            except Exception as e:
                print(f"Warning: Could not create interaction for {col1} and {col2}: {e}")
    
    return df

def prepare_train_test(df, target_col='TARGET', test_size=0.2, random_state=42):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    try:
        # Try to convert to pandas using pyarrow
        df_pd = df.to_pandas()
    except ModuleNotFoundError:
        # If pyarrow is not available, use a different approach
        print("PyArrow not found. Using alternative conversion method...")
        # Convert to pandas via numpy
        
        # Create a dictionary of columns
        data_dict = {}
        for col in df.columns:
            data_dict[col] = df[col].to_numpy()
        
        # Create pandas DataFrame from dict
        df_pd = pd.DataFrame(data_dict)
    
    # Convert object columns to category
    for col in df_pd.columns:
        if df_pd[col].dtype == 'object':
            df_pd[col] = df_pd[col].astype('category')
    
    # Split features and target
    if target_col in df_pd.columns:
        X = df_pd.drop(columns=[target_col])
        y = df_pd[target_col]
    else:
        print(f"Warning: Target column '{target_col}' not found. Using all columns as features.")
        X = df_pd
        y = np.zeros(len(df_pd))  # Dummy target
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) > 1 else None
    )
    
    # Check class imbalance
    print(f"Class distribution in training set: {np.bincount(y_train)}")
    
    # Apply SMOTE to handle class imbalance if available
    if 'SMOTE_AVAILABLE' in globals() and SMOTE_AVAILABLE:
        try:
            print("Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Class distribution: {np.bincount(y_train_resampled)}")
            print(f"Training set: {X_train_resampled.shape}, Test set: {X_test.shape}")
            return X_train_resampled, X_test, y_train_resampled, y_test
        except Exception as e:
            print(f"SMOTE failed: {e}. Proceeding with original data.")
    else:
        print("SMOTE not available. Proceeding with imbalanced data.")
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_lightgbm_model(X_train, y_train, X_test, y_test):
    # Identify categorical columns
    categorical_features = [col for col, dtype in X_train.dtypes.items() 
                           if dtype.name == 'category']
    
    print(f"Number of categorical features: {len(categorical_features)}")
    
    # Feature selection to remove irrelevant features
    print("Performing feature selection...")
    try:
        # Convert categorical features to numeric for feature selection
        X_train_fs = X_train.copy()
        X_test_fs = X_test.copy()
        
        for col in categorical_features:
            X_train_fs[col] = X_train_fs[col].cat.codes
            X_test_fs[col] = X_test_fs[col].cat.codes
        
        # Select top k features
        selector = SelectKBest(f_classif, k=min(50, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train_fs, y_train)
        X_test_selected = selector.transform(X_test_fs)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X_train.columns[selected_indices].tolist()
        
        print(f"Selected {len(selected_features)} features")
        
        # Use only selected features
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        
        # Update categorical features list
        categorical_features = [col for col in categorical_features if col in selected_features]
    except Exception as e:
        print(f"Feature selection failed: {e}. Using all features.")
    
    # Define LightGBM parameters with a wider range for hyperparameter tuning
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 500],
        'num_leaves': [31, 50, 100],
        'max_depth': [5, 8, 12],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 1.0, 5.0]
    }
    
    # Use a smaller grid for faster execution
    small_param_grid = {
        'learning_rate': [0.05, 0.1],
        'n_estimators': [200, 500],
        'num_leaves': [31, 50],
        'max_depth': [5, 8],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    # Base model
    base_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        random_state=42,
        verbose=-1
    )
    
    # Cross-validation
    print("Performing cross-validation with grid search...")
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=small_param_grid,  # Use small grid for faster execution
            cv=cv,
            scoring='roc_auc',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train, categorical_feature=categorical_features)
        
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Train final model with best parameters
        final_model = lgb.LGBMClassifier(
            **best_params,
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            random_state=42,
            verbose=20
        )
    except Exception as e:
        print(f"Grid search failed: {e}. Using default parameters.")
        # Default parameters if grid search fails
        final_model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            learning_rate=0.05,
            num_leaves=31,
            max_depth=8,
            n_estimators=500,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=20
        )
    
    # Train the final model
    print("Training final model...")
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        categorical_feature=categorical_features,
        callbacks=[lgb.early_stopping(50)]  # Increased patience
    )
    
    # Evaluate
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc_score:.4f}")
    
    return final_model, y_pred_proba

def evaluate_model(model, X_test, y_test, y_pred_proba):
    # ROC curve
    from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Convert probabilities to binary predictions using optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', 
                label=f'Optimal threshold: {optimal_threshold:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_names = X_test.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importance')
    plt.gca().invert_yaxis()
    plt.savefig('feature_importance.png')
    
    return importance_df

def main():
    # Load data
    df = load_data('train_df.csv')  # Replace with your actual file path
    
    # Check if TARGET column exists, if not, add a dummy one for testing
    if 'TARGET' not in df.columns:
        print("Warning: TARGET column not found. Adding a dummy TARGET column for testing.")
        # Add a dummy target column (0 for most rows, 1 for ~10% of rows)
        import numpy as np
        np.random.seed(42)
        target = np.random.choice([0, 1], size=df.height, p=[0.9, 0.1])
        df = df.with_columns(pl.Series(name="TARGET", values=target))
    
    # Explore data
    df = explore_data(df)
    
    # Apply the improved preprocessing
    df = preprocess_data(df)
    
    # Apply additional feature engineering
    df = feature_engineering(df)
    
    # Save preprocessed data
    df.write_csv('preprocessed_data.csv')
    print("Preprocessed data saved to preprocessed_data.csv")
    
    # Prepare train/test sets
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    
    # Train and evaluate model
    model, y_pred_proba = train_lightgbm_model(X_train, y_train, X_test, y_test)
    importance_df = evaluate_model(model, X_test, y_test, y_pred_proba)
    
    # Save top features
    importance_df.to_csv('feature_importance.csv', index=False)
    
    # Save model
    import joblib
    joblib.dump(model, 'credit_risk_model.pkl')
    print("Model saved to credit_risk_model.pkl")

if __name__ == "__main__":
    main()
