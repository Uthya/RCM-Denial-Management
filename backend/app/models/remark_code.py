from sqlalchemy import BigInteger, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin


class RemarkCode(TimestampMixin, Base):
    __tablename__ = "remark_codes"

    remittance_claim_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("remittance_claims.id"), index=True, nullable=False
    )
    remark_code: Mapped[str] = mapped_column(String(10), nullable=False)
    raw_lq_segment: Mapped[str | None] = mapped_column(Text)

    remittance_claim: Mapped["RemittanceClaim"] = relationship(
        back_populates="remark_codes"
    )
