from datetime import date, datetime
from decimal import Decimal

from app.models.enums import ClaimStatus
from app.schemas.base import SchemaBase
from app.schemas.claim_line import ClaimLineResponse
from app.schemas.diagnosis import DiagnosisResponse
from app.schemas.remittance_claim import RemittanceClaimResponse


class ClaimCreate(SchemaBase):
    claim_number: str
    payer_name: str | None = None
    patient_member_id: str | None = None
    total_charge_amount: Decimal
    facility_type_code: str | None = None
    frequency_code: str | None = None
    service_from_date: date
    service_to_date: date | None = None
    claim_status: ClaimStatus = ClaimStatus.submitted
    edi_file_id: int | None = None
    raw_claim_segment: str | None = None
    previous_payer_claim_control_no: str | None = None
    authorization_number: str | None = None
    referral_number: str | None = None
    billing_provider_npi: str | None = None
    rendering_provider_npi: str | None = None


class ClaimResponse(SchemaBase):
    id: int
    claim_number: str
    payer_name: str | None = None
    patient_member_id: str | None = None
    total_charge_amount: Decimal
    facility_type_code: str | None = None
    frequency_code: str | None = None
    service_from_date: date
    service_to_date: date | None = None
    claim_status: ClaimStatus
    edi_file_id: int | None = None
    previous_payer_claim_control_no: str | None = None
    authorization_number: str | None = None
    referral_number: str | None = None
    billing_provider_npi: str | None = None
    rendering_provider_npi: str | None = None
    created_at: datetime
    updated_at: datetime


class ClaimDetailResponse(ClaimResponse):
    claim_lines: list[ClaimLineResponse] = []
    diagnoses: list[DiagnosisResponse] = []
    remittance_claims: list[RemittanceClaimResponse] = []
