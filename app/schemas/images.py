from typing import Optional, Dict, List

from pydantic import BaseModel
from uuid import UUID
from datetime import datetime

class ImageBase(BaseModel):
    claim_id: UUID
    storage_url: str
    exif_data: Optional[Dict]
    gps_location: Optional[str]
    is_gps_mismatch: Optional[bool]
    is_metadata_missing: Optional[bool]
    vlm_summary: Optional[str]
    vlm_anomalies: Optional[str]

class ImageCreate(ImageBase):
    pass

class ImageOut(ImageCreate):
    image_id: UUID
    upload_time: datetime

    class Config:
        orm_mode = True

class DetectedPartOut(BaseModel):
    part_name: str
    bbox: List[float]
    confidence_score: float

# Detected damage schema
class DetectedDamageOut(BaseModel):
    damage_type: str
    severity_level: str
    confidence_score: float
    bbox: List[float]
    associated_part: Optional[str]

# Full response schema including parts and damages
class ImageFullOut(ImageOut):
    detected_parts: List[DetectedPartOut]
    detected_damages: List[DetectedDamageOut]