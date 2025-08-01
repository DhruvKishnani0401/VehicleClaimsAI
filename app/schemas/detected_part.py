from pydantic import BaseModel
from uuid import UUID
from typing import Dict

class DetectedPartBase(BaseModel):
    image_id: UUID
    part_name: str
    bbox: Dict
    confidence_score: float

class DetectedPartCreate(DetectedPartBase):
    pass

class DetectedPartOut(DetectedPartBase):
    part_id: UUID

    class Config:
        orm_mode = True