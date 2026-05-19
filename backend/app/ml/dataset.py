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
from app.models.enums import ClaimStatus

# Statuses that have a ground-truth adjudication outcome.
_LABELED_STATUSES = {
    ClaimStatus.paid,
    ClaimStatus.partially_paid,
    ClaimStatus.denied,
}


def _label_claim(status: ClaimStatus) -> int | None:
    """Map a claim status to a binary denial label.

    Returns 0 for paid/partially_paid, 1 for denied, None for excluded statuses.
    """
    if status == ClaimStatus.denied:
        return 1
    if status in (ClaimStatus.paid, ClaimStatus.partially_paid):
        return 0
    return None


async def build_dataset(db: AsyncSession) -> pd.DataFrame:
    """Query the DB and return a labeled DataFrame for ML training.

    Only includes claims that:
    - Have a resolved status (paid, partially_paid, denied)
    - Have at least one remittance_claim (proof of adjudication)

    Returns a DataFrame with one row per claim and a ``denied`` label column.
    """
    stmt = (
        select(Claim)
        .where(Claim.claim_status.in_(_LABELED_STATUSES))
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

        label = _label_claim(claim.claim_status)
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
