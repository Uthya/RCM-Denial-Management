from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "RCM Denial Management System"
    DEBUG: bool = False

    DATABASE_URL: str = "postgresql+asyncpg://postgres:Uthaya%4000@localhost:5433/rcm_denials"

    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    ML_MODEL_PATH: str = "app/ml/artifacts/model.json"
    ML_ENCODERS_PATH: str = "app/ml/artifacts/feature_encoders.joblib"
    ML_METRICS_PATH: str = "app/ml/artifacts/training_metrics.json"
    ML_DISTRIBUTIONS_PATH: str = "app/ml/artifacts/training_distributions.json"
    ML_CALIBRATOR_PATH: str = "app/ml/artifacts/calibrator.joblib"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
