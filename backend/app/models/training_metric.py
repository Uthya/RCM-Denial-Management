from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.base import TimestampMixin


class TrainingMetric(TimestampMixin, Base):
    __tablename__ = "model_training_metrics"
    __table_args__ = (
        Index(
            "ix_model_training_metrics_timestamp_desc",
            "training_timestamp",
        ),
    )

    training_id: Mapped[str] = mapped_column(
        String(36), unique=True, index=True, nullable=False
    )
    training_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    total_claims_used: Mapped[int] = mapped_column(Integer, nullable=False)
    training_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    test_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    denied_claims: Mapped[int] = mapped_column(Integer, nullable=False)
    paid_claims: Mapped[int] = mapped_column(Integer, nullable=False)
    denial_rate: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False)
    roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    training_time_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str | None] = mapped_column(
        String(50), nullable=True, index=True
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="success"
    )
