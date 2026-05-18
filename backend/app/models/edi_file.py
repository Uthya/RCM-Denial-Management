from sqlalchemy import Enum, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin
from app.models.enums import FileType


class EdiFile(TimestampMixin, Base):
    __tablename__ = "edi_files"

    file_type: Mapped[FileType] = mapped_column(Enum(FileType), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    interchange_control_no: Mapped[str | None] = mapped_column(String(20))
    sender_id: Mapped[str | None] = mapped_column(String(50))
    receiver_id: Mapped[str | None] = mapped_column(String(50))
    raw_text: Mapped[str | None] = mapped_column(Text)
    parser_version: Mapped[str | None] = mapped_column(String(20))

    claims: Mapped[list["Claim"]] = relationship(back_populates="edi_file")
    raw_segments: Mapped[list["RawSegment"]] = relationship(back_populates="edi_file")
