from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from app.schemas import detected_part
from app.models.detected_part import DetectedPart
from app.database import get_db

router = APIRouter(prefix="/detected-parts", tags=["Detected Parts"])

@router.post("/", response_model=detected_part.DetectedPartOut)
def create_detected_part(part: detected_part.DetectedPartCreate, db: Session = Depends(get_db)):
    db_part = DetectedPart(**part.dict())
    db.add(db_part)
    db.commit()
    db.refresh(db_part)
    return db_part

@router.get("/{part_id}", response_model=detected_part.DetectedPartOut)
def get_detected_part(part_id: UUID, db: Session = Depends(get_db)):
    db_part = db.query(DetectedPart).filter(DetectedPart.part_id == part_id).first()
    if not db_part:
        raise HTTPException(status_code=404, detail="Detected part not found")
    return db_part