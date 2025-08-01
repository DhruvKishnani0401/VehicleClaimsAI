from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from geoalchemy2 import Geography
from sqlalchemy.sql import func
import uuid

from app.database import Base

class Image(Base):
    __tablename__ = 'images'
    image_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.claim_id"), nullable=False)
    storage_url = Column(Text, nullable=False)
    upload_time = Column(DateTime(timezone=True), server_default=func.now())
    exif_data = Column(JSONB)
    gps_location = Column(Geography(geometry_type='POINT', srid=4326))
    is_gps_mismatch = Column(Boolean)
    is_metadata_missing = Column(Boolean)
    vlm_summary = Column(Text)
    vlm_anomalies = Column(Text)