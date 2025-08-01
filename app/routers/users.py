from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from app.schemas import users
from app import models
from app.database import get_db
from app.utils import hash_password

router = APIRouter(prefix="/users", tags=["users"])
@router.post("/", response_model=users.UserOut)
def create_user(user: users.UserCreate, db: Session = Depends(get_db)):
    hashed_pw = hash_password(user.password)
    db_user = models.users.User(email=user.email, role=user.role, password_hash=hashed_pw)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/{user_id}", response_model=users.UserOut)
def get_user(user_id: UUID, db: Session = Depends(get_db)):
    db_user = db.query(models.users.User).filter(models.users.User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user