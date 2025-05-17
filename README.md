# Home Credit Risk Prediction Model

This project aims to predict the credit risk of loan applicants using the Home Credit dataset. The goal is to identify clients who are likely to repay their loans, helping to ensure that deserving applicants are not rejected.

## Project Structure

- `credit_risk_model.py`: Main script for data loading, preprocessing, feature engineering, and model training
- `model_tuning.py`: Script for hyperparameter tuning and cross-validation
- `feature_analysis.py`: Script for analyzing feature importance and distributions

## Getting Started

### Prerequisites

- Python 3.12
- Required packages: polars, pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn

### Data

The model uses the Home Credit dataset. Place the following files in the project directory:
- `application_train.csv`: Main training data

### Running the Project

1. Data preprocessing and initial model training:
```
python credit_risk_model.py
```

2. Model tuning with cross-validation:
```
python model_tuning.py
```

3. Feature analysis:
```
python feature_analysis.py
```

## Model Details

- Algorithm: LightGBM (Gradient Boosting)
- Evaluation Metric: ROC-AUC
- Feature Engineering: Credit-to-income ratio, age calculation, etc.

## Results

After running the scripts, you'll find:
- Trained model: `credit_risk_model.pkl` and `best_credit_risk_model.pkl`
- Feature importance: `feature_importance.csv`
- Visualizations: ROC curve, feature importance plots, etc.

## Next Steps

- Ensemble multiple models for better performance
- Implement more advanced feature engineering
- Deploy the model as an API