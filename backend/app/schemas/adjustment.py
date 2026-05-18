from datetime import datetime
from decimal import Decimal

from app.schemas.base import SchemaBase


class AdjustmentCreate(SchemaBase):
    remittance_claim_id: int
    adjustment_group_code: str
    adjustment_reason_code: str
    adjustment_amount: Decimal
    raw_cas_segment: str | None = None


class AdjustmentResponse(SchemaBase):
    id: int
    remittance_claim_id: int
    adjustment_group_code: str
    adjustment_reason_code: str
    adjustment_amount: Decimal
    created_at: datetime
    updated_at: datetime
