"""Dataset construction for claim-level denial prediction.

Queries the database, applies labeling rules, and produces a labeled
pandas DataFrame ready for feature engineering and model training.
"""

from decimal import Decimal

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.claim import Claim

# CLP02 status codes from 835 remittance segments.
_DENIED_CLP02 = {"4"}
_PAID_CLP02 = {"1", "2", "3", "19", "20"}


def _label_from_remittances(remittance_claims: list) -> int | None:
    """First-pass denial label derived from remittance CLP02 codes.

    If ANY remittance_claim has CLP02="4" the original submission was denied,
    even if a later resubmission was paid.  Returns 1 for denied, 0 for paid,
    None when no actionable CLP02 code is present.
    """
    if not remittance_claims:
        return None
    codes = {rc.claim_status_code for rc in remittance_claims}
    if codes & _DENIED_CLP02:
        return 1  # denied on first pass
    if codes & _PAID_CLP02:
        return 0  # paid on first pass
    return None


async def build_dataset(db: AsyncSession) -> pd.DataFrame:
    """Query the DB and return a labeled DataFrame for ML training.

    Only includes claims that:
    - Are original submissions (frequency_code IS NULL or '1')
    - Have at least one remittance_claim (proof of adjudication)

    Labels are derived from remittance CLP02 codes, not the mutable
    ``claim.claim_status`` field (which gets overwritten by resubmission 835s).

    Returns a DataFrame with one row per claim and a ``denied`` label column.
    """
    stmt = (
        select(Claim)
        .where(
            (Claim.frequency_code.is_(None)) | (Claim.frequency_code == "1")
        )
        .options(
            selectinload(Claim.claim_lines),
            selectinload(Claim.diagnoses),
            selectinload(Claim.remittance_claims),
        )
    )

    result = await db.execute(stmt)
    claims = result.scalars().all()

    rows: list[dict] = []
    for claim in claims:
        # Skip claims without remittance data — no adjudication proof.
        if not claim.remittance_claims:
            continue

        label = _label_from_remittances(claim.remittance_claims)
        if label is None:
            continue

        # Sort lines/diagnoses by their sequence for deterministic "first" picks.
        sorted_lines = sorted(claim.claim_lines, key=lambda l: l.line_number)
        sorted_diagnoses = sorted(claim.diagnoses, key=lambda d: d.sequence_number)

        rows.append(
            {
                "claim_id": claim.id,
                "claim_number": claim.claim_number,
                "denied": label,
                "total_charge_amount": float(claim.total_charge_amount),
                "payer_name": claim.payer_name,
                "facility_type_code": claim.facility_type_code,
                "frequency_code": claim.frequency_code,
                "service_from_date": claim.service_from_date,
                "service_to_date": claim.service_to_date,
                "line_count": len(claim.claim_lines),
                "diagnosis_count": len(claim.diagnoses),
                "primary_diagnosis_code": (
                    sorted_diagnoses[0].diagnosis_code if sorted_diagnoses else None
                ),
                "primary_procedure_code": (
                    sorted_lines[0].procedure_code if sorted_lines else None
                ),
                "total_billed_amount": float(
                    sum(
                        (l.billed_amount for l in claim.claim_lines),
                        Decimal(0),
                    )
                ),
                "total_units": float(
                    sum(
                        (l.units for l in claim.claim_lines),
                        Decimal(0),
                    )
                ),
                "has_modifier": any(
                    l.modifier1 is not None for l in claim.claim_lines
                ),
                "place_of_service": (
                    sorted_lines[0].place_of_service if sorted_lines else None
                ),
            }
        )

    return pd.DataFrame(rows)


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Return label distribution stats for a labeled dataset DataFrame."""
    total = len(df)
    if total == 0:
        return {"total": 0, "denied": 0, "paid": 0, "denial_rate": 0.0}

    denied = int(df["denied"].sum())
    paid = total - denied
    return {
        "total": total,
        "denied": denied,
        "paid": paid,
        "denial_rate": round(denied / total, 4),
    }
