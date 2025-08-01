from sqlalchemy import Column, String, DateTime, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from app.database import Base

class FraudFlag(Base):
    __tablename__ = 'fraud_flags'
    flag_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.image_id"), nullable=False)
    flag_type = Column(String(100))
    flag_confidence = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())