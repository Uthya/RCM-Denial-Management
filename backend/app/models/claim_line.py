from datetime import date
from decimal import Decimal

from sqlalchemy import BigInteger, Date, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin


class ClaimLine(TimestampMixin, Base):
    __tablename__ = "claim_lines"
    __table_args__ = (
        Index("ix_claim_lines_claim_id_line_number", "claim_id", "line_number", unique=True),
    )

    claim_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("claims.id"), index=True, nullable=False
    )
    line_number: Mapped[int] = mapped_column(Integer, nullable=False)
    procedure_code: Mapped[str] = mapped_column(String(10), nullable=False)
    modifier1: Mapped[str | None] = mapped_column(String(5))
    modifier2: Mapped[str | None] = mapped_column(String(5))
    units: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, default=1)
    billed_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    service_date: Mapped[date | None] = mapped_column(Date)
    place_of_service: Mapped[str | None] = mapped_column(String(2))
    raw_sv1_segment: Mapped[str | None] = mapped_column(Text)

    claim: Mapped["Claim"] = relationship(back_populates="claim_lines")
