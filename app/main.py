from fastapi import FastAPI
from app.routers import claims
app = FastAPI()

app.include_router(claims.router)