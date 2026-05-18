from datetime import date
from decimal import Decimal

from sqlalchemy import BigInteger, Date, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin


class RemittanceClaim(TimestampMixin, Base):
    __tablename__ = "remittance_claims"

    claim_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("claims.id"), index=True, nullable=False
    )
    claim_status_code: Mapped[str] = mapped_column(String(10), nullable=False)
    billed_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    paid_amount: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False, default=0
    )
    payer_claim_control_number: Mapped[str | None] = mapped_column(String(50))
    remittance_date: Mapped[date] = mapped_column(Date, nullable=False)
    raw_clp_segment: Mapped[str | None] = mapped_column(Text)

    claim: Mapped["Claim"] = relationship(back_populates="remittance_claims")
    adjustments: Mapped[list["Adjustment"]] = relationship(
        back_populates="remittance_claim", cascade="all, delete-orphan"
    )
    remark_codes: Mapped[list["RemarkCode"]] = relationship(
        back_populates="remittance_claim", cascade="all, delete-orphan"
    )
