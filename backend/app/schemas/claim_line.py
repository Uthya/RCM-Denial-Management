from datetime import date, datetime
from decimal import Decimal

from app.schemas.base import SchemaBase


class ClaimLineCreate(SchemaBase):
    claim_id: int
    line_number: int
    procedure_code: str
    modifier1: str | None = None
    modifier2: str | None = None
    units: Decimal = Decimal("1")
    billed_amount: Decimal
    service_date: date | None = None
    place_of_service: str | None = None
    raw_sv1_segment: str | None = None


class ClaimLineResponse(SchemaBase):
    id: int
    claim_id: int
    line_number: int
    procedure_code: str
    modifier1: str | None = None
    modifier2: str | None = None
    units: Decimal
    billed_amount: Decimal
    service_date: date | None = None
    place_of_service: str | None = None
    created_at: datetime
    updated_at: datetime
