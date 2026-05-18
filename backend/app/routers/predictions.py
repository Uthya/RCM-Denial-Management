from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.ml.dataset import build_dataset, get_dataset_stats

router = APIRouter()


@router.post("/predict")
async def predict_denial():
    return {"message": "Denial prediction endpoint"}


@router.get("/feature-meta")
async def feature_meta():
    from app.ml.feature_engineering import FeatureEngineer
    return [vars(m) for m in FeatureEngineer.get_feature_metadata()]


@router.get("/dataset-stats")
async def dataset_stats(db: AsyncSession = Depends(get_db)):
    """Return label distribution stats for the training dataset."""
    df = await build_dataset(db)
    return get_dataset_stats(df)
