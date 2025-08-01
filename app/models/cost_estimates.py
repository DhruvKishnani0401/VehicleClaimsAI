from sqlalchemy import Column, DateTime, ForeignKey, Text, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from app.database import Base

class CostEstimate(Base):
    __tablename__ = 'cost_estimates'

    estimate_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.claim_id"), nullable=False)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.image_id"), nullable=False)
    estimated_cost_ai = Column(Numeric(10, 2))
    estimated_cost_human = Column(Numeric(10, 2))
    explanation = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())