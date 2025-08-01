from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from app.schemas import detected_damage
from app.models.detected_damage import DetectedDamage
from app.database import get_db

router = APIRouter(prefix="/detected_damage", tags=["detected damage"])

@router.post("/", response_model=detected_damage.DetectedDamageOut)
def create_damage(damage:detected_damage.DetectedDamageCreate, db: Session = Depends(get_db)):
    db_damage = DetectedDamage(**damage.dict())
    db.add(db_damage)
    db.commit()
    db.refresh(db_damage)
    return db_damage

@router.get("/{damage_id}", response_model=detected_damage.DetectedDamageOut)
def get_damage(damage_id:UUID, db: Session = Depends(get_db)):
    db_damage = db.query(DetectedDamage).filter(DetectedDamage.damage_id == damage_id).first()
    if not db_damage:
        raise HTTPException(status_code=404, detail="Damage not found")
    return db_damage