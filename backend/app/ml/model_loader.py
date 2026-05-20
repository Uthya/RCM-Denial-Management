import os

from xgboost import XGBClassifier

from app.core.config import settings


def load_model() -> XGBClassifier | None:
    """Load a trained XGBoost model from the artifacts directory."""
    model_path = settings.ML_MODEL_PATH
    if not os.path.exists(model_path):
        return None
    model = XGBClassifier()
    model.load_model(model_path)
    return model
