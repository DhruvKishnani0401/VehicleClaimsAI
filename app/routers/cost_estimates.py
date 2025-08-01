from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from app.schemas import cost_estimates
from app.models.cost_estimates import CostEstimate
from app.database import get_db

router = APIRouter(prefix="/cost_estimates", tags=["cost estimates"])

@router.post("/", response_model=cost_estimates.CostEstimateOut)
def create_cost_estimate(cost_estimate: cost_estimates.CostEstimateCreate, db: Session = Depends(get_db)):
    db_estimate = CostEstimate(**cost_estimate.dict())
    db.add(db_estimate)
    db.commit()
    db.refresh(db_estimate)
    return db_estimate

@router.get("/{estimate_id}", response_model=cost_estimates.CostEstimateOut)
def get_cost_estimate(estimate_id: UUID, db: Session = Depends(get_db)):
    db_estimate = db.query(CostEstimate).filter(CostEstimate.estimate_id == estimate_id).first()
    if not db_estimate:
        raise HTTPException(status_code=404, detail="Estimation not found")
    return db_estimate