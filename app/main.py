from fastapi import FastAPI
from app.routers import claims, users, images, detected_damage
app = FastAPI()

app.include_router(claims.router)
app.include_router(users.router)
app.include_router(images.router)
app.include_router(detected_damage.router)