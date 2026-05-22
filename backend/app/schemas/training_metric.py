from datetime import datetime

from app.schemas.base import SchemaBase


class TrainingMetricResponse(SchemaBase):
    id: int
    training_id: str
    training_timestamp: datetime
    total_claims_used: int
    training_samples: int
    test_samples: int
    denied_claims: int
    paid_claims: int
    denial_rate: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None = None
    training_time_seconds: float
    model_version: str | None = None
    notes: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime


class TrainingHistoryPage(SchemaBase):
    items: list[TrainingMetricResponse]
    total: int
    skip: int
    limit: int
