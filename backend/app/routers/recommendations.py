"""Recommendation endpoints.

- POST /by-file/{edi_file_id} : per-upload fixes for the left panel
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.claim import Claim
from app.models.claim_lifecycle import ClaimLifecycle
from app.models.edi_file import EdiFile
from app.models.enums import FileType
from app.models.raw_segment import RawSegment
from app.models.remittance_claim import RemittanceClaim
from app.services.recommendations import (
    DENIED_CLP02,
    PAID_CLP02,
    build_recommendations_for_claim,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ValidationErrorIn(BaseModel):
    segment: str
    field: str | None = None
    message: str
    severity: str | None = None
    claim_identifier: str | None = None
    validator: str | None = None


class RecommendationsRequest(BaseModel):
    validation_errors: list[ValidationErrorIn] | None = None


class RecommendationItem(BaseModel):
    reason: str
    fix: str
    source: Literal["parser", "carc", "model"]
    location: str | None = None


class ClaimRecommendation(BaseModel):
    claim_id: int
    claim_number: str
    payer_name: str | None = None
    status_badge: Literal["Denied", "High Risk", "Resolved"]
    risk_score: float | None = None
    resolved: bool = False
    recommendations: list[RecommendationItem]


class ByFileResponse(BaseModel):
    edi_file_id: int
    file_type: str
    total_claims_in_file: int
    flagged_claims: int
    claims: list[ClaimRecommendation]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _badge_for_837_claim(risk_level: str | None) -> str | None:
    """Badge for claims appearing in the 837 (submission) panel.

    The 837 panel is the submission-time / prediction-driven view. It only
    surfaces HIGH risk; MEDIUM is suppressed for biller focus and "Denied"
    is intentionally NOT possible here even when the claim happens to
    already carry claim_status=denied from a prior 835 — that label belongs
    in the 835 panel. A claim that has been rectified by a paid replacement
    is upgraded to "Resolved" by the caller after this function runs.
    """
    if risk_level == "HIGH":
        return "High Risk"
    return None


RESOLVED_RECOMMENDATION: dict = {
    "reason": "Replacement claim adjudicated as paid",
    "fix": (
        "Fix verified — the replacement claim was paid by the payer. "
        "No further action required."
    ),
    "source": "carc",
    "location": "Resolved",
}


async def _resolved_claim_ids(
    db: AsyncSession, candidate_ids: list[int]
) -> set[int]:
    """Of the given claim ids, return those whose denial has been rectified
    by a successfully-paid replacement claim.

    Resolution criteria:
      1. The claim has a paid remittance directly (rare for denials, but covers
         the case where the same claim was reprocessed and paid).
      2. A replacement claim linked via claim_lifecycles has a paid remittance.
      3. A claim with the same claim_number AND frequency_code in {'6','7'}
         has a paid remittance — catches the case where REF*F8/lifecycle
         linkage is missing but the resubmission is otherwise identifiable.
    """
    if not candidate_ids:
        return set()

    resolved: set[int] = set()

    # 1. Direct same-claim payment
    direct_paid = (
        select(RemittanceClaim.claim_id)
        .where(RemittanceClaim.claim_id.in_(candidate_ids))
        .where(RemittanceClaim.claim_status_code.in_(PAID_CLP02))
        .distinct()
    )
    resolved.update((await db.execute(direct_paid)).scalars().all())

    # 2. Paid replacement via lifecycle
    via_lifecycle = (
        select(ClaimLifecycle.original_claim_id)
        .join(
            RemittanceClaim,
            RemittanceClaim.claim_id == ClaimLifecycle.child_claim_id,
        )
        .where(ClaimLifecycle.original_claim_id.in_(candidate_ids))
        .where(RemittanceClaim.claim_status_code.in_(PAID_CLP02))
        .distinct()
    )
    resolved.update((await db.execute(via_lifecycle)).scalars().all())

    # 3. Paid replacement via claim_number match (lifecycle-independent
    # fallback). Use the candidates' own claim_numbers to find sibling
    # claims with freq IN {6,7} that have been paid.
    unresolved = [cid for cid in candidate_ids if cid not in resolved]
    if unresolved:
        OrigClaim = Claim.__table__.alias("orig")
        RepClaim = Claim.__table__.alias("rep")
        via_number = (
            select(OrigClaim.c.id)
            .join(RepClaim, RepClaim.c.claim_number == OrigClaim.c.claim_number)
            .join(
                RemittanceClaim,
                RemittanceClaim.claim_id == RepClaim.c.id,
            )
            .where(OrigClaim.c.id.in_(unresolved))
            .where(RepClaim.c.id != OrigClaim.c.id)
            .where(RepClaim.c.frequency_code.in_(("6", "7")))
            .where(RemittanceClaim.claim_status_code.in_(PAID_CLP02))
            .distinct()
        )
        resolved.update((await db.execute(via_number)).scalars().all())

    return resolved


def _claim_to_predict_dict(claim: Claim) -> dict:
    """Mirror predictions.router._claim_to_predict_dict to avoid the cross-router import."""
    from decimal import Decimal

    lines = claim.claim_lines or []
    diagnoses = claim.diagnoses or []
    sorted_diags = sorted(diagnoses, key=lambda d: d.sequence_number)
    sorted_lines = sorted(lines, key=lambda ln: ln.line_number)
    total_billed = (
        float(sum((ln.billed_amount for ln in lines), Decimal(0)))
        if lines
        else float(claim.total_charge_amount or 0)
    )
    total_units = (
        float(sum((ln.units for ln in lines), Decimal(0))) if lines else 1.0
    )
    return {
        "total_charge_amount": float(claim.total_charge_amount or 0),
        "payer_name": claim.payer_name,
        "facility_type_code": claim.facility_type_code,
        "frequency_code": claim.frequency_code,
        "service_from_date": (
            str(claim.service_from_date) if claim.service_from_date else None
        ),
        "service_to_date": (
            str(claim.service_to_date) if claim.service_to_date else None
        ),
        "line_count": len(lines) or 1,
        "diagnosis_count": len(diagnoses) or 1,
        "primary_diagnosis_code": (
            sorted_diags[0].diagnosis_code if sorted_diags else None
        ),
        "primary_procedure_code": (
            sorted_lines[0].procedure_code if sorted_lines else None
        ),
        "total_billed_amount": total_billed,
        "total_units": total_units or 1.0,
        "has_modifier": any(ln.modifier1 or ln.modifier2 for ln in lines),
        "place_of_service": sorted_lines[0].place_of_service if sorted_lines else None,
    }


# ---------------------------------------------------------------------------
# Endpoint: by-file (left panel)
# ---------------------------------------------------------------------------

@router.post("/by-file/{edi_file_id}", response_model=ByFileResponse)
async def recommendations_by_file(
    edi_file_id: int,
    body: RecommendationsRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Return per-claim recommendations for the most recently uploaded file.

    Behaviour by file type:
    - 837: predicts each claim; flags HIGH/MEDIUM risk; includes parser
      findings (if the caller passes them in) and ML risk factors.
    - 835: identifies which claims this remittance file touched; for any that
      adjudicated as denied (CLP02=4), builds CARC-based recommendations
      from each claim's adjustments.
    """
    # Look up the file
    edi = (
        await db.execute(select(EdiFile).where(EdiFile.id == edi_file_id))
    ).scalar_one_or_none()
    if not edi:
        raise HTTPException(status_code=404, detail=f"EDI file {edi_file_id} not found")

    if edi.file_type == FileType.edi_837:
        return await _recommendations_for_837(edi, body, db)
    if edi.file_type == FileType.edi_835:
        return await _recommendations_for_835(edi, db)

    raise HTTPException(status_code=400, detail=f"Unsupported file type: {edi.file_type}")


async def _recommendations_for_837(
    edi: EdiFile,
    body: RecommendationsRequest | None,
    db: AsyncSession,
) -> ByFileResponse:
    # Validation errors arrive grouped by claim_identifier (claim_number)
    parser_findings_by_claim: dict[str, list[dict]] = {}
    if body and body.validation_errors:
        for ve in body.validation_errors:
            if ve.validator == "payer":
                # 835-style CAS findings — not relevant for an 837 upload context
                continue
            key = ve.claim_identifier or ""
            parser_findings_by_claim.setdefault(key, []).append(
                {
                    "segment": ve.segment,
                    "field": ve.field,
                    "message": ve.message,
                }
            )

    # Pull all claims in this file with the data the predictor needs
    stmt = (
        select(Claim)
        .where(Claim.edi_file_id == edi.id)
        .options(selectinload(Claim.claim_lines), selectinload(Claim.diagnoses))
    )
    claims = (await db.execute(stmt)).scalars().all()
    total_claims = len(claims)

    # Predictor (optional — degrades gracefully if model isn't trained yet)
    try:
        from app.ml.predictor import get_predictor

        predictor = get_predictor()
        predictor_ready = predictor.is_ready
    except Exception:
        predictor = None
        predictor_ready = False

    flagged: list[ClaimRecommendation] = []
    for claim in claims:
        risk_score: float | None = None
        risk_level: str | None = None
        ml_factors: list[dict] = []

        if predictor_ready:
            try:
                pred = predictor.predict(_claim_to_predict_dict(claim))
                risk_score = float(pred["risk_score"])
                risk_level = pred["risk_level"]
                ml_factors = pred.get("top_risk_factors") or []
            except Exception:
                logger.exception("Prediction failed for claim %s", claim.claim_number)

        badge = _badge_for_837_claim(risk_level)
        # Surface only HIGH-risk claims in the 837 panel
        if badge is None:
            continue

        parser_findings = parser_findings_by_claim.get(claim.claim_number, [])

        recs = build_recommendations_for_claim(
            parser_findings=parser_findings,
            carc_entries=None,  # 837 has no payer adjudication yet
            ml_factors=ml_factors,
        )

        if not recs:
            # No actionable content — skip rather than show an empty card
            continue

        flagged.append(
            ClaimRecommendation(
                claim_id=claim.id,
                claim_number=claim.claim_number,
                payer_name=claim.payer_name,
                status_badge=badge,
                risk_score=risk_score,
                recommendations=[RecommendationItem(**r) for r in recs],
            )
        )

    # Mark resolved claims: any flagged claim whose denial has been rectified
    # by a paid replacement gets badge="Resolved" and a single success rec.
    resolved_ids = await _resolved_claim_ids(db, [c.claim_id for c in flagged])
    for c in flagged:
        if c.claim_id in resolved_ids:
            c.status_badge = "Resolved"
            c.resolved = True
            c.recommendations = [RecommendationItem(**RESOLVED_RECOMMENDATION)]

    # Order: Denied → High Risk → Resolved, then by risk_score desc
    badge_order = {"Denied": 0, "High Risk": 1, "Resolved": 2}
    flagged.sort(
        key=lambda c: (
            badge_order.get(c.status_badge, 99),
            -(c.risk_score or 0),
        )
    )

    return ByFileResponse(
        edi_file_id=edi.id,
        file_type=edi.file_type.value,
        total_claims_in_file=total_claims,
        flagged_claims=len(flagged),
        claims=flagged,
    )


async def _recommendations_for_835(
    edi: EdiFile,
    db: AsyncSession,
) -> ByFileResponse:
    # Find remittance_claims that were created by this 835 file. The link is
    # via raw_clp_segment text == raw_segments.raw_segment_text where
    # raw_segments.edi_file_id is the 835's id and segment_name='CLP'.
    rc_stmt = (
        select(RemittanceClaim)
        .join(
            RawSegment,
            RawSegment.raw_segment_text == RemittanceClaim.raw_clp_segment,
        )
        .where(RawSegment.edi_file_id == edi.id)
        .where(RawSegment.segment_name == "CLP")
        .options(
            selectinload(RemittanceClaim.adjustments),
            selectinload(RemittanceClaim.remark_codes),
            selectinload(RemittanceClaim.claim),
        )
    )
    remittances = (await db.execute(rc_stmt)).scalars().unique().all()

    flagged: list[ClaimRecommendation] = []
    for rc in remittances:
        # Only interested in denied adjudications for the recommendations view
        if rc.claim_status_code not in DENIED_CLP02:
            continue
        if rc.claim is None:
            continue

        carc_entries = [
            {
                "reason_code": adj.adjustment_reason_code,
                "group_code": adj.adjustment_group_code,
            }
            for adj in (rc.adjustments or [])
        ]

        recs = build_recommendations_for_claim(carc_entries=carc_entries)
        if not recs:
            continue

        flagged.append(
            ClaimRecommendation(
                claim_id=rc.claim.id,
                claim_number=rc.claim.claim_number,
                payer_name=rc.claim.payer_name,
                status_badge="Denied",
                risk_score=None,
                recommendations=[RecommendationItem(**r) for r in recs],
            )
        )

    # Mark resolved denials: if a paid replacement exists for the claim,
    # swap the badge to "Resolved" and replace the CARC fixes with a single
    # success message — the biller has already fixed it.
    resolved_ids = await _resolved_claim_ids(db, [c.claim_id for c in flagged])
    for c in flagged:
        if c.claim_id in resolved_ids:
            c.status_badge = "Resolved"
            c.resolved = True
            c.recommendations = [RecommendationItem(**RESOLVED_RECOMMENDATION)]

    # Surface unresolved denials first, then resolved
    flagged.sort(key=lambda c: (0 if c.status_badge == "Denied" else 1))

    return ByFileResponse(
        edi_file_id=edi.id,
        file_type=edi.file_type.value,
        total_claims_in_file=len(remittances),
        flagged_claims=len(flagged),
        claims=flagged,
    )
