from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional

class FraudFlagBase(BaseModel):
    image_id: UUID
    flag_type: str
    flag_confidence: float

class FraudFlagCreate(FraudFlagBase):
    pass

class FraudFlagOut(FraudFlagBase):
    flag_id: UUID
    created_at: datetime

    class Config:
        orm_mode = True