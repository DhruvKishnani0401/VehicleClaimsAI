from fastapi import FastAPI
from app.routers import claims
from app.routers import users, images


app = FastAPI()

app.include_router(claims.router, prefix="/claims", tags=["claims"])
app.include_router(users.router, prefix="/users", tags=["users"])

app.include_router(images.router, prefix="/images", tags=["images"])