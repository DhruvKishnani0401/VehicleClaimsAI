from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime

class ClaimBase(BaseModel):
    user_id: UUID
    incident_reported_at: datetime
    status: Optional[str] = "Pending"

class ClaimCreate(ClaimBase):
    pass

class ClaimOut(ClaimBase):
    claim_id: UUID
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True