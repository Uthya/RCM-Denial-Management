"""Best-effort persistence of every prediction the model makes.

This service is intentionally fault-tolerant: a DB failure here MUST NOT
break the prediction API. Callers wrap log_prediction in try/except.

Keys stored in feature_snapshot are the RAW inputs to the predictor, not
the engineered numeric vector. Raw inputs are stable across retrainings;
encoded values are not. See the monitoring plan for rationale.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.prediction_log import PredictionLog

logger = logging.getLogger(__name__)


_SNAPSHOT_FIELDS: tuple[str, ...] = (
    "payer_name",
    "primary_procedure_code",
    "primary_diagnosis_code",
    "place_of_service",
    "facility_type_code",
    "frequency_code",
    "total_charge_amount",
    "total_billed_amount",
    "total_units",
    "line_count",
    "diagnosis_count",
    "has_modifier",
    "service_from_date",
    "service_to_date",
)


def _build_snapshot(claim_input: dict) -> dict:
    """Pick the documented subset of raw inputs to persist as JSONB."""
    snap: dict = {}
    for key in _SNAPSHOT_FIELDS:
        if key in claim_input:
            value = claim_input[key]
            # Dates may come in as datetime.date; JSONB can't store those.
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            snap[key] = value
    return snap


def _label_from_risk(risk_score: float) -> int:
    return 1 if risk_score >= 0.5 else 0


def _resolve_label(prediction_result: dict, risk_score: float) -> int:
    """Prefer the predictor's label (tuned threshold); fall back to 0.5.

    Newer predictions carry ``predicted_label`` computed at the model's tuned
    decision threshold. Older results (pre-calibration) don't, so we derive it
    from the 0.5 rule to stay backward compatible.
    """
    label = prediction_result.get("predicted_label")
    if label is not None:
        return int(label)
    return _label_from_risk(risk_score)


def _row_from_prediction(
    *,
    claim_id: int | None,
    claim_number: str | None,
    claim_input: dict,
    prediction_result: dict,
) -> PredictionLog:
    """Pure builder — does not touch the DB."""
    risk_score = float(prediction_result["risk_score"])
    pred_time = prediction_result.get("prediction_timestamp")
    if pred_time:
        try:
            prediction_time = datetime.fromisoformat(pred_time)
        except ValueError:
            prediction_time = datetime.utcnow()
    else:
        prediction_time = datetime.utcnow()

    return PredictionLog(
        claim_id=claim_id,
        claim_number=claim_number,
        prediction_id=prediction_result["prediction_id"],
        predicted_risk=risk_score,
        predicted_label=_resolve_label(prediction_result, risk_score),
        risk_level=prediction_result["risk_level"],
        model_version=prediction_result.get("model_version", "unknown"),
        feature_engineering_version=prediction_result.get(
            "feature_version", "unknown"
        ),
        feature_snapshot=_build_snapshot(claim_input),
        prediction_time=prediction_time,
    )


async def log_prediction(
    db: AsyncSession,
    *,
    claim_id: int | None,
    claim_number: str | None,
    claim_input: dict,
    prediction_result: dict,
) -> None:
    """Insert one prediction_log row. Swallows any error after logging it."""
    try:
        row = _row_from_prediction(
            claim_id=claim_id,
            claim_number=claim_number,
            claim_input=claim_input,
            prediction_result=prediction_result,
        )
        db.add(row)
        await db.commit()
    except Exception:
        logger.exception("Failed to persist prediction_log row")
        try:
            await db.rollback()
        except Exception:
            logger.exception("Rollback after prediction_log failure also failed")


async def log_predictions_bulk(
    db: AsyncSession,
    entries: list[dict],
) -> None:
    """Bulk-insert variant for /predict-file. Each entry has keys:
    claim_id, claim_number, claim_input, prediction_result.
    """
    if not entries:
        return
    try:
        rows = [
            _row_from_prediction(
                claim_id=e["claim_id"],
                claim_number=e["claim_number"],
                claim_input=e["claim_input"],
                prediction_result=e["prediction_result"],
            )
            for e in entries
        ]
        db.add_all(rows)
        await db.commit()
    except Exception:
        logger.exception("Bulk prediction_log insert failed (%d rows)", len(entries))
        try:
            await db.rollback()
        except Exception:
            logger.exception("Rollback after bulk prediction_log failure also failed")
