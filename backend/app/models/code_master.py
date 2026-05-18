from sqlalchemy import Enum, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.base import TimestampMixin
from app.models.enums import CodeType


class CodeMaster(TimestampMixin, Base):
    __tablename__ = "code_masters"
    __table_args__ = (
        Index("ix_code_masters_type_code", "code_type", "code", unique=True),
    )

    code_type: Mapped[CodeType] = mapped_column(Enum(CodeType), nullable=False)
    code: Mapped[str] = mapped_column(String(10), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
