from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Float, Numeric
from sqlalchemy.dialects.postgresql import UUID, VARCHAR, JSONB
from geoalchemy2 import Geography
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

from app.database import Base

class Claim(Base):
    __tablename__ = 'claims'
    claim_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True),ForeignKey("users.user_id"),  nullable=False)
    incident_reported_at = Column(DateTime, nullable=False)
    reported_location = Column(Geography(geometry_type='POINT', srid=4326))
    status = Column(VARCHAR(50), default="Pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
