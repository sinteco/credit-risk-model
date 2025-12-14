from fastapi import FastAPI
from .pydantic_models import CreditScoringRequest, CreditScoringResponse

app = FastAPI()

@app.post("/predict", response_model=CreditScoringResponse)
def predict(request: CreditScoringRequest):
    # TODO: Implement prediction endpoint
    return {"score": 0.0, "risk_level": "low"}
