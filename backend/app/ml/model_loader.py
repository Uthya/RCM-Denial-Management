import os
import xgboost as xgb

from app.core.config import settings


def load_model() -> xgb.Booster | None:
    """Load a trained XGBoost model from the artifacts directory."""
    model_path = settings.ML_MODEL_PATH
    if not os.path.exists(model_path):
        return None
    model = xgb.Booster()
    model.load_model(model_path)
    return model
