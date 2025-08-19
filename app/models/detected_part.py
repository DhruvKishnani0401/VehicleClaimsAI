from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Float, Numeric
from sqlalchemy.dialects.postgresql import UUID, VARCHAR, JSONB
from geoalchemy2 import Geography
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from app.database import Base

class DetectedPart(Base):
    __tablename__ = 'detected_parts'
    part_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.image_id"), nullable=False)
    part_name = Column(String(255), nullable=False)
    bbox = Column(JSONB)
    confidence_score = Column(Float)

    image = relationship("Image", back_populates="detected_parts")