from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "RCM Denial Management System"
    DEBUG: bool = False

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rcm_denials"

    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    ML_MODEL_PATH: str = "app/ml/artifacts/model.json"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
