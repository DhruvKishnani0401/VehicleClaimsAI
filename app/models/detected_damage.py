from sqlalchemy import Column, String, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from app.database import Base

class DetectedDamage(Base):
    __tablename__ = 'detected_damages'

    damage_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.image_id"), nullable=False)
    damage_type = Column(String(100))
    severity_level = Column(String(50))
    segmentation_mask = Column(JSONB)
    associated_part = Column(String(100))
    confidence_score = Column(Float)