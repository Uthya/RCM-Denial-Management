from __future__ import annotations

import logging
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.ml.dataset import build_dataset, get_dataset_stats
from app.models.claim import Claim
from app.services.prediction_logger import log_prediction, log_predictions_bulk

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    total_charge_amount: float
    payer_name: str | None = None
    facility_type_code: str | None = None
    frequency_code: str | None = None
    service_from_date: date
    service_to_date: date | None = None
    line_count: int
    diagnosis_count: int
    primary_diagnosis_code: str | None = None
    primary_procedure_code: str | None = None
    total_billed_amount: float
    total_units: float
    has_modifier: bool = False
    place_of_service: str | None = None
    # v5 inputs — all optional so older clients keep working. Authorization /
    # referral numbers feed presence flags; provider NPIs feed target-encoded
    # features; submission_date defaults to today (the request timestamp).
    authorization_number: str | None = None
    referral_number: str | None = None
    billing_provider_npi: str | None = None
    rendering_provider_npi: str | None = None
    submission_date: date | None = None


class RiskFactor(BaseModel):
    feature: str
    impact: str
    direction: str


class UnseenIndicators(BaseModel):
    """Per-claim flag: which (if any) categorical fields were not in the
    model's training vocabulary. ``any`` is True when any of the tracked
    dimensions is True.

    ``billing_provider`` and ``rendering_provider`` are v5 additions and
    are optional on the response so an older v4-trained model can still
    return a valid payload (the engineer's vocabulary won't contain those
    keys and the predictor simply omits the extra fields)."""

    payer: bool
    cpt: bool
    dx: bool
    any: bool
    billing_provider: bool | None = None
    rendering_provider: bool | None = None


class PredictionResponse(BaseModel):
    prediction_id: str
    risk_score: float
    risk_level: str
    top_risk_factors: list[RiskFactor]
    model_version: str
    feature_version: str
    prediction_timestamp: str
    # Optional v3.3+ fields (omitted by older models so the schema stays
    # backward compatible).
    raw_risk_score: float | None = None
    predicted_label: int | None = None
    decision_threshold: float | None = None
    unseen_indicators: UnseenIndicators | None = None


class ClaimPrediction(BaseModel):
    claim_id: int
    claim_number: str
    payer_name: str | None
    total_charge_amount: float
    risk_score: float
    risk_level: str
    top_risk_factors: list[RiskFactor]
    # True iff any of payer/CPT/Dx for this claim was not in the model's
    # training vocabulary. Used by the upload UI to badge "new data" claims.
    unseen_any: bool | None = None


class FilePredictionResponse(BaseModel):
    edi_file_id: int
    total_claims: int
    predicted_claims: int
    failed_claims: int
    average_risk_score: float
    risk_summary: dict[str, int]
    claims: list[ClaimPrediction]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claim_to_predict_dict(claim: Claim) -> dict:
    """Build the raw dict for the predictor from a Claim ORM object."""
    lines = claim.claim_lines or []
    diagnoses = claim.diagnoses or []

    line_count = len(lines)
    diagnosis_count = len(diagnoses)
    total_billed = float(sum(ln.billed_amount for ln in lines)) if lines else float(claim.total_charge_amount or 0)
    total_units = float(sum(ln.units for ln in lines)) if lines else 1.0
    has_modifier = any(ln.modifier1 or ln.modifier2 for ln in lines)
    place_of_service = lines[0].place_of_service if lines else None

    # Sort by sequence/line number to get "primary"
    sorted_diags = sorted(diagnoses, key=lambda d: d.sequence_number)
    primary_diag = sorted_diags[0].diagnosis_code if sorted_diags else None

    sorted_lines = sorted(lines, key=lambda ln: ln.line_number)
    primary_proc = sorted_lines[0].procedure_code if sorted_lines else None

    # v5: submission_date is the date the 837 was ingested into our system.
    # ``claim.created_at`` is set by TimestampMixin in the same transaction
    # as the EdiFile row, so it's the cleanest available "when did we get
    # this claim" timestamp without an extra join. Strictly precedes any
    # 835/denial, so it cannot leak future information into training.
    submission_date = (
        claim.created_at.date()
        if getattr(claim, "created_at", None) is not None
        else None
    )

    return {
        "total_charge_amount": float(claim.total_charge_amount or 0),
        "payer_name": claim.payer_name,
        "facility_type_code": claim.facility_type_code,
        "frequency_code": claim.frequency_code,
        "service_from_date": str(claim.service_from_date) if claim.service_from_date else None,
        "service_to_date": str(claim.service_to_date) if claim.service_to_date else None,
        "line_count": line_count or 1,
        "diagnosis_count": diagnosis_count or 1,
        "primary_diagnosis_code": primary_diag,
        "primary_procedure_code": primary_proc,
        "total_billed_amount": total_billed,
        "total_units": total_units or 1.0,
        "has_modifier": has_modifier,
        "place_of_service": place_of_service,
        # v5 inputs from the persisted claim. The feature engineer derives
        # has_prior_authorization / has_referral from presence and uses the
        # NPIs as TargetEncoder inputs (with their own OOV flags).
        "authorization_number": claim.authorization_number,
        "referral_number": claim.referral_number,
        "billing_provider_npi": claim.billing_provider_npi,
        "rendering_provider_npi": claim.rendering_provider_npi,
        "submission_date": str(submission_date) if submission_date else None,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict-claim/{claim_id}", response_model=PredictionResponse)
async def predict_claim_by_id(claim_id: int, db: AsyncSession = Depends(get_db)):
    """Run denial prediction for a single claim by its database ID."""
    from app.ml.predictor import get_predictor

    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train the model first via POST /api/predictions/train.",
        )

    stmt = (
        select(Claim)
        .where(Claim.id == claim_id)
        .options(selectinload(Claim.claim_lines), selectinload(Claim.diagnoses))
    )
    row = await db.execute(stmt)
    claim = row.scalars().first()

    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

    try:
        pred_input = _claim_to_predict_dict(claim)
        result = predictor.predict(pred_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    await log_prediction(
        db,
        claim_id=claim.id,
        claim_number=claim.claim_number,
        claim_input=pred_input,
        prediction_result=result,
    )

    return result


@router.post("/predict", response_model=PredictionResponse)
async def predict_denial(
    request: PredictionRequest, db: AsyncSession = Depends(get_db)
):
    from app.ml.predictor import get_predictor

    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train the model first via POST /api/predictions/train.",
        )

    # For ad-hoc predictions the "submission date" is when this request is
    # being made — the operator is asking "if I submit this claim today,
    # what's the denial risk?" Default to today's date when the client did
    # not supply one explicitly.
    sub_date = request.submission_date or date.today()

    claim = {
        "total_charge_amount": request.total_charge_amount,
        "payer_name": request.payer_name,
        "facility_type_code": request.facility_type_code,
        "frequency_code": request.frequency_code,
        "service_from_date": str(request.service_from_date) if request.service_from_date else None,
        "service_to_date": str(request.service_to_date) if request.service_to_date else None,
        "line_count": request.line_count,
        "diagnosis_count": request.diagnosis_count,
        "primary_diagnosis_code": request.primary_diagnosis_code,
        "primary_procedure_code": request.primary_procedure_code,
        "total_billed_amount": request.total_billed_amount,
        "total_units": request.total_units,
        "has_modifier": request.has_modifier,
        "place_of_service": request.place_of_service,
        # v5 inputs
        "authorization_number": request.authorization_number,
        "referral_number": request.referral_number,
        "billing_provider_npi": request.billing_provider_npi,
        "rendering_provider_npi": request.rendering_provider_npi,
        "submission_date": str(sub_date),
    }

    try:
        result = predictor.predict(claim)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    await log_prediction(
        db,
        claim_id=None,
        claim_number=None,
        claim_input=claim,
        prediction_result=result,
    )

    return result


@router.post("/predict-file/{edi_file_id}", response_model=FilePredictionResponse)
async def predict_file(edi_file_id: int, db: AsyncSession = Depends(get_db)):
    """Run denial predictions on all claims from an uploaded EDI file."""
    from app.ml.predictor import get_predictor

    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train the model first via POST /api/predictions/train.",
        )

    stmt = (
        select(Claim)
        .where(Claim.edi_file_id == edi_file_id)
        .options(selectinload(Claim.claim_lines), selectinload(Claim.diagnoses))
    )
    rows = await db.execute(stmt)
    claims = rows.scalars().all()

    if not claims:
        raise HTTPException(status_code=404, detail=f"No claims found for edi_file_id={edi_file_id}")

    # Vectorized prediction: one FE.transform + one model.predict_proba +
    # one calibrator + one tree-SHAP call for the entire file. The previous
    # implementation called predictor.predict(...) per-claim, paying ~40 ms
    # of pandas fixed cost per row; on 1000 claims that was ~88s, vs ~1s
    # for the batched path. Per-row outputs are byte-identical (see the
    # equivalence verification harness).
    #
    # Pre-build the claim dicts in the same order as the ORM rows so we
    # can zip results back into ClaimPrediction objects.
    pred_inputs: list[dict] = [_claim_to_predict_dict(claim) for claim in claims]

    failed = 0
    results: list[ClaimPrediction] = []
    log_entries: list[dict] = []
    try:
        preds = predictor.predict_batch(pred_inputs)
    except Exception:
        # A batch-level failure means FE / model / calibrator broke for
        # the whole file (schema mismatch, etc.). Surface it as the
        # caller-visible 500 rather than silently zeroing every claim.
        logger.exception(
            "Batched prediction failed for edi_file_id=%s (%d claims)",
            edi_file_id,
            len(claims),
        )
        raise HTTPException(
            status_code=500,
            detail="Batched prediction failed; check server logs.",
        )

    for claim, pred_input, pred in zip(claims, pred_inputs, preds, strict=True):
        unseen = pred.get("unseen_indicators") or {}
        results.append(ClaimPrediction(
            claim_id=claim.id,
            claim_number=claim.claim_number,
            payer_name=claim.payer_name,
            total_charge_amount=float(claim.total_charge_amount or 0),
            risk_score=pred["risk_score"],
            risk_level=pred["risk_level"],
            top_risk_factors=pred["top_risk_factors"],
            unseen_any=bool(unseen.get("any")) if unseen else None,
        ))
        log_entries.append({
            "claim_id": claim.id,
            "claim_number": claim.claim_number,
            "claim_input": pred_input,
            "prediction_result": pred,
        })

    await log_predictions_bulk(db, log_entries)

    # Sort by risk score descending (highest risk first)
    results.sort(key=lambda r: r.risk_score, reverse=True)

    scores = [r.risk_score for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    risk_summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in results:
        risk_summary[r.risk_level] += 1

    return FilePredictionResponse(
        edi_file_id=edi_file_id,
        total_claims=len(claims),
        predicted_claims=len(results),
        failed_claims=failed,
        average_risk_score=round(avg_score, 4),
        risk_summary=risk_summary,
        claims=results,
    )


@router.get("/feature-meta")
async def feature_meta():
    from app.ml.feature_engineering import FeatureEngineer
    return [vars(m) for m in FeatureEngineer.get_feature_metadata()]


@router.get("/dataset-stats")
async def dataset_stats(db: AsyncSession = Depends(get_db)):
    """Return label distribution stats for the training dataset.

    Server-side aggregate (was pulling 670k claims + 2.8M lines +
    2.5M diagnoses + 670k remits across the WAN to count in pandas;
    over an 11 Mbps link that took 30+ minutes). The query below
    replicates the labeling rule from ``_label_from_remittances``
    exactly:
      - only originals (frequency_code NULL or '1')
      - need at least one remit
      - 'denied' if ANY remit has CLP02='4'; else 'paid' if ANY has
        CLP02 in (1,2,3,19,20); else excluded (label=None)
    The whole thing runs in Postgres and returns 4 numbers, sub-second
    even over WAN.
    """
    from sqlalchemy import text
    sql = text("""
        WITH labeled AS (
            SELECT c.id,
                   CASE
                       WHEN bool_or(rc.claim_status_code = '4') THEN 1
                       WHEN bool_or(rc.claim_status_code IN ('1','2','3','19','20')) THEN 0
                       ELSE NULL
                   END AS denied
            FROM claims c
            JOIN remittance_claims rc ON rc.claim_id = c.id
            WHERE c.frequency_code IS NULL OR c.frequency_code = '1'
            GROUP BY c.id
        )
        SELECT
            count(*) FILTER (WHERE denied IS NOT NULL)::bigint AS total,
            count(*) FILTER (WHERE denied = 1)::bigint AS denied,
            count(*) FILTER (WHERE denied = 0)::bigint AS paid
        FROM labeled
    """)
    row = (await db.execute(sql)).one()
    total = int(row.total or 0)
    denied = int(row.denied or 0)
    paid = int(row.paid or 0)
    denial_rate = round(denied / total, 4) if total else 0.0
    return {
        "total": total,
        "denied": denied,
        "paid": paid,
        "denial_rate": denial_rate,
    }


@router.post("/train")
async def train_denial_model(
    db: AsyncSession = Depends(get_db),
    mode: str = Query(
        "full",
        regex="^(full|warm|quick)$",
        description=(
            "Retrain mode. 'full' = Optuna search + isotonic calibration "
            "(use after feature changes or scheduled retrains). "
            "'warm' = reuse last good hyperparameters, skip Optuna entirely "
            "(routine retrains; falls back to full if no compatible prior "
            "artifact). 'quick' = warm + 3-fold calibration (fastest)."
        ),
    ),
    tune: bool = Query(
        True,
        description="Tune hyperparameters with Optuna (full mode only).",
    ),
    n_trials: int | None = Query(
        None,
        ge=1,
        le=200,
        description=(
            "Maximum Optuna trials. Omit for an adaptive budget based on "
            "dataset size (5/10/20/30 for <500/<5k/<50k/>=50k rows)."
        ),
    ),
    tuning_timeout: int | None = Query(
        180,
        ge=1,
        description="Wall-clock budget for tuning in seconds.",
    ),
):
    from app.ml.trainer import train_model
    from app.ml.predictor import get_predictor
    try:
        result = await train_model(
            db,
            tune=tune,
            n_trials=n_trials,
            tuning_timeout=tuning_timeout,
            mode=mode,
        )
        # Reload the singleton predictor so it picks up the new artifacts
        get_predictor().load()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
