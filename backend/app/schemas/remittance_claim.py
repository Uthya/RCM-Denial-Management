from datetime import date, datetime
from decimal import Decimal

from app.schemas.adjustment import AdjustmentResponse
from app.schemas.base import SchemaBase
from app.schemas.remark_code import RemarkCodeResponse


class RemittanceClaimCreate(SchemaBase):
    claim_id: int
    claim_status_code: str
    billed_amount: Decimal
    paid_amount: Decimal = Decimal("0")
    payer_claim_control_number: str | None = None
    remittance_date: date
    raw_clp_segment: str | None = None


class RemittanceClaimResponse(SchemaBase):
    id: int
    claim_id: int
    claim_status_code: str
    billed_amount: Decimal
    paid_amount: Decimal
    payer_claim_control_number: str | None = None
    remittance_date: date
    created_at: datetime
    updated_at: datetime
    adjustments: list[AdjustmentResponse] = []
    remark_codes: list[RemarkCodeResponse] = []
