from fastapi import FastAPI
from app.routers import claims, users, images, detected_part, detected_damage, cost_estimates, fraud_flags
app = FastAPI()

app.include_router(claims.router)
app.include_router(users.router)
app.include_router(images.router)
app.include_router(detected_part.router)
app.include_router(detected_damage.router)
app.include_router(cost_estimates.router)
app.include_router(fraud_flags.router)