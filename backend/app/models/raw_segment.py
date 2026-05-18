from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin


class RawSegment(TimestampMixin, Base):
    __tablename__ = "raw_segments"
    __table_args__ = (
        Index("ix_raw_segments_edi_file_position", "edi_file_id", "segment_position"),
    )

    edi_file_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("edi_files.id"), index=True
    )
    claim_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("claims.id")
    )
    segment_name: Mapped[str] = mapped_column(String(10), nullable=False)
    segment_position: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_segment_text: Mapped[str] = mapped_column(Text, nullable=False)
    parse_error: Mapped[str | None] = mapped_column(Text)

    edi_file: Mapped["EdiFile | None"] = relationship(back_populates="raw_segments")
    claim: Mapped["Claim | None"] = relationship(back_populates="raw_segments")
