from datetime import datetime
import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from uuid import UUID

from app.AI_funcs.AI_model import detect_damages
from app.schemas import images
from app.models.images import Image
from app.models.detected_damage import DetectedDamage
from app.models.detected_part import DetectedPart
from app.database import get_db
from app.AI_funcs import AI_model


UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
router = APIRouter(prefix="/images", tags=["images"])

@router.post("/", response_model=images.ImageFullOut)
async def upload_image(claim_id: UUID = Form(...), file:UploadFile = File(...), db: Session = Depends(get_db)):
    allowed_extensions = ["jpg", "jpeg", "png", "heic"]
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    new_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, new_filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    db_image = Image(claim_id=claim_id, storage_url=file_path, upload_time=datetime.utcnow())
    db.add(db_image)
    db.commit()
    db.refresh(db_image)

    parts = await AI_model.detect_damages(file_path)
    types = await AI_model.detect_types(file_path)

    for p in parts:
        db.add(DetectedPart(image_id=db_image.image_id, **p))

    for t in types:
        db.add(DetectedDamage(image_id=db_image.image_id, **t))

    db.commit()
    db.refresh(db_image)
    return images.ImageFullOut(
        **db_image.__dict__,
        detected_parts=parts,
        detected_damages=types
    )

@router.get("/{image_id}", response_model=images.ImageOut)
def get_image(image_id: UUID, db: Session = Depends(get_db)):
    db_image = db.query(Image).filter(Image.image_id == image_id).first()
    if not db_image:
        raise HTTPException(status_code=404, detail="Image not found")
    return db_image

