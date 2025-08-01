from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from app import models
from app.schemas import claims
from app.database import get_db

router = APIRouter(prefix="/claims", tags=["claims"])

@router.post("/", response_model=claims.ClaimOut)
def create_claim(claim: claims.ClaimCreate, db: Session = Depends(get_db)):
    db_claim = models.claims.Claim(**claim.dict())
    db.add(db_claim)
    db.commit()
    db.refresh(db_claim)
    return db_claim

@router.get("/{claims_id}", response_model=claims.ClaimOut)
def get_claim(claims_id: UUID, db: Session = Depends(get_db)):
    db_claim = db.query(models.claims.Claim).filter(models.claims.Claim.claim_id == claims_id).first()
    if not db_claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return db_claim