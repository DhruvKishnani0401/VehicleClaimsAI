from typing import Optional
from pydantic import BaseModel, EmailStr
from uuid import UUID
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    role: str

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    user_id: UUID
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        orm_mode = True