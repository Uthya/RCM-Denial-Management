from app.ml.dataset import build_dataset
from app.ml.feature_engineering import FeatureEngineer, extract_features, FEATURE_COLUMNS

def train_model(*args, **kwargs):
    from app.ml.trainer import train_model as _train_model
    return _train_model(*args, **kwargs)
