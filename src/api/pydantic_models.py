from pydantic import BaseModel

class CreditScoringRequest(BaseModel):
    # TODO: Define input fields
    transaction_id: str
    amount: float

class CreditScoringResponse(BaseModel):
    score: float
    risk_level: str
