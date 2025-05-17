import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.inspection import permutation_importance

def load_model_and_data():
    """Load the trained model and preprocessed data"""
    model = joblib.load('credit_risk_model.pkl')
    df = pl.read_csv('preprocessed_data.csv')
    return model, df

def analyze_feature_importance(model, X):
    """Analyze and visualize feature importance"""
    # Get feature importance from model
    feature_importance = model.feature_importances_
    feature_names = X.columns
    
    # Create DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('top_features.png')
    
    return importance_df

def plot_feature_distributions(df, top_features, target_col='TARGET'):
    """Plot distributions of top features by target class"""
    df_pd = df.to_pandas()
    
    for feature in top_features[:5]:  # Plot top 5 features
        plt.figure(figsize=(10, 6))
        
        # Check if feature is categorical or continuous
        if df_pd[feature].nunique() < 10:  # Categorical
            # Create a crosstab
            ct = pd.crosstab(df_pd[feature], df_pd[target_col])
            ct_pct = ct.div(ct.sum(axis=1), axis=0)
            ct_pct.plot(kind='bar', stacked=True)
            plt.title(f'Distribution of {feature} by Target')
            plt.ylabel('Percentage')
        else:  # Continuous
            # Create KDE plot
            sns.kdeplot(data=df_pd, x=feature, hue=target_col, common_norm=False)
            plt.title(f'Distribution of {feature} by Target')
            
        plt.tight_layout()
        plt.savefig(f'feature_dist_{feature}.png')
        plt.close()

def permutation_feature_importance(model, X, y):
    """Calculate permutation feature importance"""
    # Convert to numpy arrays for permutation importance
    X_np = X.values
    y_np = y.values
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_np, y_np, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    # Create DataFrame for easier manipulation
    perm_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    # Plot permutation importance
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=perm_importance_df.head(20))
    plt.title('Permutation Feature Importance (Top 20)')
    plt.tight_layout()
    plt.savefig('permutation_importance.png')
    
    return perm_importance_df

def analyze_correlations(df, top_features):
    """Analyze correlations between top features"""
    df_pd = df.to_pandas()
    
    # Select top features and target
    selected_cols = top_features[:15].tolist()  # Top 15 features
    if 'TARGET' in df_pd.columns:
        selected_cols.append('TARGET')
    
    # Calculate correlation matrix
    corr_matrix = df_pd[selected_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Top Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    return corr_matrix

def main():
    # Load model and data
    model, df = load_model_and_data()
    
    # Convert to pandas for analysis
    df_pd = df.to_pandas()
    X = df_pd.drop(columns=['TARGET'])
    y = df_pd['TARGET']
    
    # Analyze feature importance
    print("Analyzing feature importance...")
    importance_df = analyze_feature_importance(model, X)
    top_features = importance_df['Feature'].tolist()
    
    # Plot feature distributions
    print("Plotting feature distributions...")
    plot_feature_distributions(df, top_features)
    
    # Calculate permutation importance
    print("Calculating permutation importance...")
    perm_importance_df = permutation_feature_importance(model, X, y)
    
    # Analyze correlations
    print("Analyzing feature correlations...")
    corr_matrix = analyze_correlations(df, top_features)
    
    # Save results
    importance_df.to_csv('model_feature_importance.csv', index=False)
    perm_importance_df.to_csv('permutation_feature_importance.csv', index=False)
    
    print("Feature analysis completed. Results saved to CSV files and plots.")

if __name__ == "__main__":
    main()