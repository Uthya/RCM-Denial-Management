"""Read-only endpoints for the model training history table."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.training_metric import TrainingMetric
from app.schemas.training_metric import TrainingHistoryPage, TrainingMetricResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/training-history", response_model=TrainingHistoryPage)
async def list_training_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Return paginated training runs, newest first."""
    total = await db.scalar(select(func.count()).select_from(TrainingMetric))
    stmt = (
        select(TrainingMetric)
        .order_by(TrainingMetric.training_timestamp.desc(), TrainingMetric.id.desc())
        .offset(skip)
        .limit(limit)
    )
    rows = await db.execute(stmt)
    items = rows.scalars().all()
    return TrainingHistoryPage(
        items=[TrainingMetricResponse.model_validate(r) for r in items],
        total=int(total or 0),
        skip=skip,
        limit=limit,
    )


@router.get("/training-history/{training_id}", response_model=TrainingMetricResponse)
async def get_training_run(training_id: str, db: AsyncSession = Depends(get_db)):
    """Return a single training run by its external ``training_id``."""
    stmt = select(TrainingMetric).where(TrainingMetric.training_id == training_id)
    row = await db.execute(stmt)
    rec = row.scalar_one_or_none()
    if rec is None:
        raise HTTPException(
            status_code=404, detail=f"Training run {training_id} not found"
        )
    return rec


@router.get("/latest-training", response_model=TrainingMetricResponse)
async def get_latest_training(db: AsyncSession = Depends(get_db)):
    """Return the most recent training run."""
    stmt = (
        select(TrainingMetric)
        .order_by(TrainingMetric.training_timestamp.desc(), TrainingMetric.id.desc())
        .limit(1)
    )
    row = await db.execute(stmt)
    rec = row.scalar_one_or_none()
    if rec is None:
        raise HTTPException(status_code=404, detail="No training runs recorded yet")
    return rec
