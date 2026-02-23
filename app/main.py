from fastapi import FastAPI
from app.api.v1.routes import router as v1_router
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Loan Default Risk API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Loan Default Risk API is running"}

app.include_router(
    v1_router,
    prefix="/api/v1",
    tags=["v1"]
)