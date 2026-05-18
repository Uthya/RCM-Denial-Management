from datetime import datetime

from app.models.enums import CodeType
from app.schemas.base import SchemaBase


class CodeMasterCreate(SchemaBase):
    code_type: CodeType
    code: str
    description: str


class CodeMasterResponse(SchemaBase):
    id: int
    code_type: CodeType
    code: str
    description: str
    created_at: datetime
    updated_at: datetime
