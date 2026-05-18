from datetime import datetime

from app.schemas.base import SchemaBase


class DiagnosisCreate(SchemaBase):
    claim_id: int
    diagnosis_code: str
    diagnosis_type: str
    sequence_number: int
    raw_hi_segment: str | None = None


class DiagnosisResponse(SchemaBase):
    id: int
    claim_id: int
    diagnosis_code: str
    diagnosis_type: str
    sequence_number: int
    created_at: datetime
    updated_at: datetime
