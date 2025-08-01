from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Float, Numeric
from sqlalchemy.dialects.postgresql import UUID, VARCHAR, JSONB
from geoalchemy2 import Geography
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class Claim(Base):
    __tablename__ = 'claims'
    claim_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True),ForeignKey("users.user_id"),  nullable=False)
    incident_reported_at = Column(DateTime, nullable=False)
    reported_location = Column(Geography(geometry_type='POINT', srid=4326))
    status = Column(VARCHAR(50), default="Pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class User(Base):
    __tablename__ = 'users'
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False, unique=True)
    password_hash = Column(Text, nullable=False)
    role = Column(VARCHAR(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))

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

class DetectedPart(Base):
    __tablename__ = 'detected_parts'
    part_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.image_id"), nullable=False)
    part_name = Column(String(255), nullable=False)
    bbox = Column(JSONB)
    confidence_score = Column(Float)

class DetectedDamage(Base):
    __tablename__ = 'detected_damages'

    damage_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.image_id"), nullable=False)
    damage_type = Column(String(100))
    severity_level = Column(String(50))
    segmentation_mask = Column(JSONB)
    associated_part = Column(String(100))
    confidence_score = Column(Float)

class CostEstimate(Base):
    __tablename__ = 'cost_estimates'

    estimate_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.claim_id"), nullable=False)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.image_id"), nullable=False)
    estimated_cost_ai = Column(Numeric(10, 2))
    estimated_cost_human = Column(Numeric(10, 2))
    explanation = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class FraudFlag(Base):
    __tablename__ = 'fraud_flags'
    flag_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.image_id"), nullable=False)
    falg_type = Column(String(100))
    flag_confidence = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

