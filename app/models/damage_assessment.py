"""
Data models for damage assessment results
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


@dataclass
class DetectedPart:
    """Represents a detected vehicle part"""
    part_name: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence_score: float


@dataclass
class DetectedDamage:
    """Represents a detected damage on a vehicle part"""
    damage_type: str  # e.g., 'dent', 'scratch', 'crack', 'shatter'
    severity_level: str  # 'minor', 'moderate', 'severe'
    segmentation_mask: Optional[List[List[int]]] = None  # For precise damage localization
    associated_part: Optional[str] = None
    confidence_score: float = 0.0
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]


@dataclass
class DamageAssessment:
    """Complete damage assessment result"""
    image_path: str
    vlm_summary: Optional[str] = None
    vlm_anomalies: Optional[List[str]] = None
    detected_parts: List[DetectedPart] = None
    detected_damages: List[DetectedDamage] = None
    estimated_cost: float = 0.0
    location_verification: Dict = None
    fraud_indicators: List[Dict] = None
    confidence_score: float = 0.0
    processing_time: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.detected_parts is None:
            self.detected_parts = []
        if self.detected_damages is None:
            self.detected_damages = []
        if self.location_verification is None:
            self.location_verification = {}
        if self.fraud_indicators is None:
            self.fraud_indicators = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()


# Pydantic models for API responses
class DetectedPartResponse(BaseModel):
    part_name: str
    bbox: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    confidence_score: float = Field(..., ge=0.0, le=1.0)


class DetectedDamageResponse(BaseModel):
    damage_type: str
    severity_level: str
    associated_part: Optional[str] = None
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    bbox: Optional[List[float]] = None


class LocationVerificationResponse(BaseModel):
    gps_match: bool
    timestamp_match: bool
    distance_km: float
    time_difference_hours: float


class FraudIndicatorResponse(BaseModel):
    type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str


class DamageAssessmentResponse(BaseModel):
    image_path: str
    vlm_summary: Optional[str] = None
    vlm_anomalies: Optional[List[str]] = None
    detected_parts: List[DetectedPartResponse] = []
    detected_damages: List[DetectedDamageResponse] = []
    estimated_cost: float = Field(..., ge=0.0)
    location_verification: LocationVerificationResponse
    fraud_indicators: List[FraudIndicatorResponse] = []
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: float = Field(..., ge=0.0)
    created_at: datetime


class ClaimImageUploadRequest(BaseModel):
    claim_id: str
    incident_location_lat: Optional[float] = None
    incident_location_lon: Optional[float] = None
    incident_time: Optional[datetime] = None
    description: Optional[str] = None


class ClaimImageUploadResponse(BaseModel):
    image_id: str
    assessment_id: str
    status: str
    message: str
    estimated_processing_time: int = Field(..., description="Estimated processing time in seconds")


class ProcessingStatusResponse(BaseModel):
    assessment_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: float = Field(..., ge=0.0, le=100.0)
    estimated_completion_time: Optional[datetime] = None
    error_message: Optional[str] = None


# Enums for type safety
class DamageType(str):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    SHATTER = "shatter"
    CRUMPLE = "crumple"
    MISSING = "missing"


class SeverityLevel(str):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class VehiclePart(str):
    BUMPER = "bumper"
    HOOD = "hood"
    DOOR = "door"
    FENDER = "fender"
    WINDSHIELD = "windshield"
    MIRROR = "mirror"
    LIGHT = "light"
    WHEEL = "wheel"
    TIRE = "tire"


class FraudIndicatorType(str):
    MISSING_GPS = "missing_gps"
    GPS_MISMATCH = "gps_mismatch"
    TIMESTAMP_MISMATCH = "timestamp_mismatch"
    IMAGE_MANIPULATION = "image_manipulation"
    INCONSISTENT_LIGHTING = "inconsistent_lighting"
    BLURRY_IMAGE = "blurry_image" 