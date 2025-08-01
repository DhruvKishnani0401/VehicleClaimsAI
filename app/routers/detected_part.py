from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from app.schemas import detected_part
from app import models
from app.database import get_db

router = APIRouter(prefix="/detected_parts", tags=["detected parts"])

@router.post("/", response_model=detected_part.DetectedPartOut)
def create_detected_part(part: detected_part.DetectedPartCreate, db: Session = Depends(get_db)):
    db_part = models.detected_part.DetectedPart(**part.dict())
    db.add(db_part)
    db.commit()
    db.refresh(db_part)
    return db_part

@router.get("/{part_id}", response_model=detected_part.DetectedPartOut)
def get_detected_part(part_id: UUID, db: Session = Depends(get_db)):
    db_part = db.query(models.detected_part.DetectedPart).filter(models.detected_part.DetectedPart.part_id == part_id)
    if not(db_part):
        raise HTTPException(status_code=404, detail="Part not found")
    return db_part