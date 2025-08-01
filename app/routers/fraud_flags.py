from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from app.schemas import fraud_flags
from app.models.fraud_flags import FraudFlag
from app.database import get_db


router = APIRouter(prefix="/fraud_flags", tags=["fraud flags"])

@router.post("/", response_model=fraud_flags.FraudFlagOut)
def create_fraud_flag(flag: fraud_flags.FraudFlagCreate, db: Session = Depends(get_db)):
    db_flag = FraudFlag(**flag.dict())
    db.add(db_flag)
    db.commit()
    db.refresh(db_flag)
    return db_flag

@router.get("/{flag_id}", response_model=fraud_flags.FraudFlagOut)
def get_fraud_flag(flag_id: UUID, db: Session = Depends(get_db)):
    db_flag = db.query(FraudFlag).filter(FraudFlag.flag_id==flag_id).first()
    if not db_flag:
        raise HTTPException(status_code=404, detail="Fraud flag not found")
    return db_flag