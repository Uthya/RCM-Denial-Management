from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.base import TimestampMixin


class PredictionLog(TimestampMixin, Base):
    __tablename__ = "prediction_log"
    __table_args__ = (
        Index(
            "ix_prediction_log_unresolved",
            "claim_id",
            "resolved_at",
        ),
        Index(
            "ix_prediction_log_perf_rollup",
            "resolved_at",
            "prediction_time",
        ),
    )

    claim_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("claims.id"), nullable=True, index=True
    )
    claim_number: Mapped[str | None] = mapped_column(
        String(50), nullable=True, index=True
    )

    prediction_id: Mapped[str] = mapped_column(
        String(36), unique=True, index=True, nullable=False
    )
    predicted_risk: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_label: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)

    model_version: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )
    feature_engineering_version: Mapped[str] = mapped_column(
        String(50), nullable=False
    )

    feature_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    prediction_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    actual_denied: Mapped[int | None] = mapped_column(
        Integer, nullable=True, index=True
    )
    actual_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    resolved_by_remittance_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("remittance_claims.id"), nullable=True
    )
