from datetime import datetime

from app.models.enums import FileType
from app.schemas.base import SchemaBase


class EdiFileCreate(SchemaBase):
    file_type: FileType
    file_name: str
    interchange_control_no: str | None = None
    sender_id: str | None = None
    receiver_id: str | None = None
    raw_text: str | None = None


class EdiFileResponse(SchemaBase):
    id: int
    file_type: FileType
    file_name: str
    interchange_control_no: str | None = None
    sender_id: str | None = None
    receiver_id: str | None = None
    created_at: datetime
    updated_at: datetime


class ValidationErrorResponse(SchemaBase):
    segment: str
    field: str
    message: str
    severity: str
    position: int = 0
    claim_identifier: str | None = None
    claim_id: int | None = None
    validator: str = ""


class ParseResultResponse(SchemaBase):
    edi_file_id: int | None = None
    file_type: str = ""
    claims_count: int = 0
    claim_lines_count: int = 0
    diagnoses_count: int = 0
    remittance_claims_count: int = 0
    adjustments_count: int = 0
    remark_codes_count: int = 0
    raw_segments_count: int = 0
    errors: list[str] = []
    success: bool = False
    validation_errors: list[ValidationErrorResponse] = []
    validation_warning_count: int = 0
    validation_error_count: int = 0
