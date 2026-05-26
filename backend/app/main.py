from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import (
    appeals,
    auth,
    claims,
    denials,
    edi,
    monitoring,
    predictions,
    recommendations,
    training_history,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="RCM Denial Management System",
    description="API for managing claim denials, appeals, and denial predictions",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(claims.router, prefix="/api/claims", tags=["Claims"])
app.include_router(denials.router, prefix="/api/denials", tags=["Denials"])
app.include_router(appeals.router, prefix="/api/appeals", tags=["Appeals"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(edi.router, prefix="/api/edi", tags=["EDI"])
app.include_router(training_history.router, prefix="/api/ml", tags=["ML"])
app.include_router(
    recommendations.router, prefix="/api/recommendations", tags=["Recommendations"]
)
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Monitoring"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
