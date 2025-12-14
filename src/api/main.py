import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from .pydantic_models import CustomerData, PredictionResponse
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

app = FastAPI(title="Credit Risk Scoring API")

# Load Model
EXPERIMENT_NAME = "Credit_Risk_Model_Experiment"
model = None

def load_best_model():
    global model
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            print(f"Experiment '{EXPERIMENT_NAME}' not found.")
            return
        
        # Find best run based on ROC_AUC
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.roc_auc DESC"])
        if runs.empty:
            print("No runs found.")
            return
            
        best_run_id = runs.iloc[0].run_id
        print(f"Best Run ID: {best_run_id}")
        
        # Find the model artifact in the run
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(best_run_id)
        if not artifacts:
             print("No artifacts found in best run.")
             return
        
        # Assuming the first artifact is the model (or the only one)
        # In train.py we logged model with name "Logistic_Regression" or "Random_Forest"
        model_artifact_path = artifacts[0].path 
        
        model_uri = f"runs:/{best_run_id}/{model_artifact_path}"
        print(f"Loading model from {model_uri}...")
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully.")

    except Exception as e:
        print(f"Error loading model: {e}")

# Load model on startup
load_best_model()

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    # Convert input to DataFrame
    input_data = pd.DataFrame([data.model_dump()])
    
    # Predict
    try:
        # The model is a pipeline, so it handles preprocessing
        prob = model.predict_proba(input_data)[:, 1][0]
        is_high_risk = int(prob > 0.5) # Threshold can be adjusted
        
        return PredictionResponse(risk_probability=prob, is_high_risk=is_high_risk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
