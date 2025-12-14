import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import load_data, build_pipeline

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using various metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }
    return metrics

def train_and_log_model(model_name, model, param_grid, X_train, y_train, X_test, y_test):
    """
    Train a model with hyperparameter tuning and log to MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        print(f"Training {model_name}...")
        
        # Hyperparameter Tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best Params for {model_name}: {best_params}")
        
        # Evaluate
        metrics = evaluate_model(best_model, X_test, y_test)
        print(f"Metrics for {model_name}: {metrics}")
        
        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, model_name)
        
        return best_model, metrics

def main():
    # Configuration
    DATA_PATH = '../data/raw/insurance.csv'
    EXPERIMENT_NAME = "Credit_Risk_Model_Experiment"
    
    # Set MLflow Experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 1. Load Data
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        print("Data file not found. Please check the path.")
        return

    # 2. Create Target Variable (Proxy)
    # Since we don't have the 'is_high_risk' from Task 4 merged physically, 
    # we recreate a proxy target here for demonstration.
    # Let's assume high charges (> median) = High Risk (1)
    df['target'] = (df['charges'] > df['charges'].median()).astype(int)
    
    # 3. Preprocessing
    numerical_features = ['age', 'bmi', 'children'] # 'charges' is used for target, so exclude
    categorical_features = ['sex', 'smoker', 'region']
    
    X = df[numerical_features + categorical_features]
    y = df['target']
    
    # Build and apply pipeline
    pipeline = build_pipeline(numerical_features, categorical_features)
    X_processed = pipeline.fit_transform(X)
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    # 5. Define Models and Hyperparameters
    models_config = [
        {
            "name": "Logistic_Regression",
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        {
            "name": "Random_Forest",
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        }
    ]
    
    best_overall_model = None
    best_overall_auc = -1
    
    # 6. Train and Evaluate
    for config in models_config:
        model, metrics = train_and_log_model(
            config["name"], 
            config["model"], 
            config["params"], 
            X_train, y_train, X_test, y_test
        )
        
        if metrics["roc_auc"] > best_overall_auc:
            best_overall_auc = metrics["roc_auc"]
            best_overall_model = model
            best_model_name = config["name"]

    print(f"\nBest Model: {best_model_name} with ROC-AUC: {best_overall_auc}")
    
    # Register Best Model (Optional - requires MLflow server setup usually, but works locally too)
    # mlflow.register_model(f"runs:/{run_id}/{best_model_name}", "Best_Credit_Risk_Model")

if __name__ == "__main__":
    main()
