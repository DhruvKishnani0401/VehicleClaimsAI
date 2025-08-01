from typing import Optional, Dict

from pydantic import BaseModel
from uuid import UUID
from datetime import datetime

class ImageBase(BaseModel):
    claim_id: UUID
    storage_url: str
    exif_data: Optional[Dict]
    gps_location: Optional[str]
    is_gps_mismatched: Optional[bool]
    is_metadata_missing: Optional[bool]
    vlm_summary: Optional[str]
    vlm_anomalies: Optional[str]

class ImageCreate(ImageBase):
    pass

class ImageOut(ImageCreate):
    image_id: UUID
    upload_time: datetime
