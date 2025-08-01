from fastapi import FastAPI
from app.routers import claims
from app.routers import users, images, detected_part


app = FastAPI()

app.include_router(claims.router)
app.include_router(users.router)

app.include_router(images.router)
app.include_router(detected_part.router)