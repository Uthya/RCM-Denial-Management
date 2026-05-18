from datetime import date
from decimal import Decimal

from sqlalchemy import BigInteger, Date, Enum, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin
from app.models.enums import ClaimStatus


class Claim(TimestampMixin, Base):
    __tablename__ = "claims"
    __table_args__ = (
        Index("ix_claims_edi_file_claim_number", "edi_file_id", "claim_number", unique=True),
    )

    claim_number: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    payer_name: Mapped[str | None] = mapped_column(String(255))
    patient_member_id: Mapped[str | None] = mapped_column(String(80), index=True)
    total_charge_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    facility_type_code: Mapped[str | None] = mapped_column(String(10))
    frequency_code: Mapped[str | None] = mapped_column(String(5))
    service_from_date: Mapped[date] = mapped_column(Date, nullable=False)
    service_to_date: Mapped[date | None] = mapped_column(Date)
    claim_status: Mapped[ClaimStatus] = mapped_column(
        Enum(ClaimStatus), nullable=False, default=ClaimStatus.submitted, index=True
    )
    edi_file_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("edi_files.id"), index=True
    )
    raw_claim_segment: Mapped[str | None] = mapped_column(Text)
    previous_payer_claim_control_no: Mapped[str | None] = mapped_column(String(50))

    edi_file: Mapped["EdiFile | None"] = relationship(back_populates="claims")
    claim_lines: Mapped[list["ClaimLine"]] = relationship(
        back_populates="claim", cascade="all, delete-orphan"
    )
    diagnoses: Mapped[list["Diagnosis"]] = relationship(
        back_populates="claim", cascade="all, delete-orphan"
    )
    remittance_claims: Mapped[list["RemittanceClaim"]] = relationship(
        back_populates="claim"
    )
    raw_segments: Mapped[list["RawSegment"]] = relationship(back_populates="claim")
