from __future__ import annotations

import logging
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.ml.dataset import build_dataset, get_dataset_stats
from app.models.claim import Claim

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


class RiskFactor(BaseModel):
    feature: str
    impact: str
    direction: str


class PredictionResponse(BaseModel):
    prediction_id: str
    risk_score: float
    risk_level: str
    top_risk_factors: list[RiskFactor]
    model_version: str
    feature_version: str
    prediction_timestamp: str


class ClaimPrediction(BaseModel):
    claim_id: int
    claim_number: str
    payer_name: str | None
    total_charge_amount: float
    risk_score: float
    risk_level: str
    top_risk_factors: list[RiskFactor]


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

    return result


@router.post("/predict", response_model=PredictionResponse)
async def predict_denial(request: PredictionRequest):
    from app.ml.predictor import get_predictor

    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train the model first via POST /api/predictions/train.",
        )

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
    }

    try:
        result = predictor.predict(claim)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

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

    results = []
    failed = 0
    for claim in claims:
        try:
            pred_input = _claim_to_predict_dict(claim)
            pred = predictor.predict(pred_input)
            results.append(ClaimPrediction(
                claim_id=claim.id,
                claim_number=claim.claim_number,
                payer_name=claim.payer_name,
                total_charge_amount=float(claim.total_charge_amount or 0),
                risk_score=pred["risk_score"],
                risk_level=pred["risk_level"],
                top_risk_factors=pred["top_risk_factors"],
            ))
        except Exception:
            logger.exception("Prediction failed for claim %s", claim.claim_number)
            failed += 1

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
    """Return label distribution stats for the training dataset."""
    df = await build_dataset(db)
    return get_dataset_stats(df)


@router.post("/train")
async def train_denial_model(db: AsyncSession = Depends(get_db)):
    from app.ml.trainer import train_model
    from app.ml.predictor import get_predictor
    try:
        result = await train_model(db)
        # Reload the singleton predictor so it picks up the new artifacts
        get_predictor().load()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
