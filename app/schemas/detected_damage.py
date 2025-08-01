from pydantic import BaseModel
from uuid import UUID
from typing import Dict

class DetectedDamageBase(BaseModel):
    image_id: UUID
    damage_type: str
    severity_level: str
    segmentation_mask: Dict
    associated_part: str
    confidence_score: float

class DetectedDamageCreate(DetectedDamageBase):
    pass

class DetectedDamageOut(DetectedDamageBase):
    pass
class DetectedDamage(BaseModel):
    damage_id: UUID

    class Config:
        orm_mode = True