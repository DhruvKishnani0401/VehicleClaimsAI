from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from geoalchemy2 import Geography
from sqlalchemy.sql import func
import uuid

from app.database import Base

class Claim(Base):
    __tablename__ = "claims"

    claim_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    status = Column(String(50), default="pending")
    incident_reported_at = Column(DateTime, nullable=False)
