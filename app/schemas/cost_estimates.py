from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional
from decimal import Decimal

class CostEstimateBase(BaseModel):
    claim_id: UUID
    image_id: UUID
    estimated_cost_ai: Optional[Decimal]
    estimated_cost_human: Optional[Decimal]
    explanation: Optional[str]

class CostEstimateCreate(CostEstimateBase):
    pass

class CostEstimateOut(CostEstimateBase):
    estimate_id: UUID
    created_at: datetime

    class Config:
        orm_mode = True
