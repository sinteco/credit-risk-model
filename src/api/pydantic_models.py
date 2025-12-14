from pydantic import BaseModel
from typing import Literal

class CustomerData(BaseModel):
    age: int
    bmi: float
    children: int
    sex: Literal['male', 'female']
    smoker: Literal['yes', 'no']
    region: Literal['southwest', 'southeast', 'northwest', 'northeast']

class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
