from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin


class Diagnosis(TimestampMixin, Base):
    __tablename__ = "diagnoses"
    __table_args__ = (
        Index("ix_diagnoses_claim_id_sequence", "claim_id", "sequence_number", unique=True),
    )

    claim_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("claims.id"), index=True, nullable=False
    )
    diagnosis_code: Mapped[str] = mapped_column(String(10), nullable=False)
    diagnosis_type: Mapped[str] = mapped_column(String(10), nullable=False)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_hi_segment: Mapped[str | None] = mapped_column(Text)

    claim: Mapped["Claim"] = relationship(back_populates="diagnoses")
