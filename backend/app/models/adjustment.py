from decimal import Decimal

from sqlalchemy import BigInteger, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin


class Adjustment(TimestampMixin, Base):
    __tablename__ = "adjustments"

    remittance_claim_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("remittance_claims.id"), index=True, nullable=False
    )
    adjustment_group_code: Mapped[str] = mapped_column(String(5), nullable=False)
    adjustment_reason_code: Mapped[str] = mapped_column(
        String(10), nullable=False, index=True
    )
    adjustment_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    # CAS triplet position 3 (CAS04, CAS07, ...) — optional per X12 005010.
    # Captures the unit/quantity adjustment when present; many 835s omit it.
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)
    raw_cas_segment: Mapped[str | None] = mapped_column(Text)

    remittance_claim: Mapped["RemittanceClaim"] = relationship(
        back_populates="adjustments"
    )
