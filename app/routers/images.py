from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from app.schemas import images
from app import models
from app.database import get_db

router = APIRouter(prefix="/images", tags=["images"])

@router.post("/", response_model=images.ImageOut)
def create_image(image: images.ImageCreate, db: Session = Depends(get_db)):
    db_image = models.images.Image(**image.dict())
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image

@router.get("/{image_id}", response_model=images.ImageOut)
def get_image(image_id: UUID, db: Session = Depends(get_db)):
    db_image = db.query(models.images.Image).filter(models.images.Image.image_id == image_id).first()
    if not db_image:
        raise HTTPException(status_code=404, detail="Image not found")
    return db_image