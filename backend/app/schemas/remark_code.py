from datetime import datetime

from app.schemas.base import SchemaBase


class RemarkCodeCreate(SchemaBase):
    remittance_claim_id: int
    remark_code: str
    raw_lq_segment: str | None = None


class RemarkCodeResponse(SchemaBase):
    id: int
    remittance_claim_id: int
    remark_code: str
    created_at: datetime
    updated_at: datetime
