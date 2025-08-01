from fastapi import FastAPI
from app.routers import claims
from app.routers import users

app = FastAPI()

app.include_router(claims.router)
app.include_router(users.router)