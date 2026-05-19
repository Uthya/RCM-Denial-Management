from sqlalchemy import BigInteger, Enum, ForeignKey, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin
from app.models.enums import RelationshipType


class ClaimLifecycle(TimestampMixin, Base):
    __tablename__ = "claim_lifecycles"
    __table_args__ = (
        Index(
            "ix_claim_lifecycles_original_child",
            "original_claim_id",
            "child_claim_id",
            unique=True,
        ),
    )

    original_claim_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("claims.id"), index=True, nullable=False
    )
    parent_claim_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("claims.id"), index=True, nullable=False
    )
    child_claim_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("claims.id"), index=True, nullable=False
    )
    relationship_type: Mapped[RelationshipType] = mapped_column(
        Enum(RelationshipType), nullable=False
    )
    iteration_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    original_claim: Mapped["Claim"] = relationship(foreign_keys=[original_claim_id])
    parent_claim: Mapped["Claim"] = relationship(foreign_keys=[parent_claim_id])
    child_claim: Mapped["Claim"] = relationship(foreign_keys=[child_claim_id])
