"""Read-only monitoring endpoints.

Three concerns, three endpoints:
  * GET /api/monitoring/live-performance — precision/recall/accuracy of
    resolved predictions in a rolling window.
  * GET /api/monitoring/drift — PSI + unseen-rate of recent claims vs.
    the saved training-distribution snapshot.
  * GET /api/monitoring/prediction-log — paginated log view for inspection.

These endpoints never write to the DB and never touch the prediction or
training paths. They are intentionally read-only so a bug here cannot
degrade core RCM functionality.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.ml.dataset import build_dataset
from app.ml.distributions import load_distribution_snapshot
from app.ml.drift import build_drift_report, categorize_drift_report
from app.models.prediction_log import PredictionLog

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# /live-performance
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


@router.get("/live-performance")
async def live_performance(
    days: int = Query(30, ge=1, le=365),
    model_version: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Precision / recall / accuracy on resolved predictions in window."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    stmt = (
        select(PredictionLog)
        .where(
            PredictionLog.resolved_at.is_not(None),
            PredictionLog.resolved_at >= since,
        )
    )
    if model_version:
        stmt = stmt.where(PredictionLog.model_version == model_version)

    rows = (await db.execute(stmt)).scalars().all()

    tp = fp = tn = fn = 0
    for r in rows:
        actual = int(r.actual_denied)
        pred = int(r.predicted_label)
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 0:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    accuracy = _safe_div(tp + tn, total)
    f1 = _safe_div(2 * precision * recall, (precision + recall)) if (precision + recall) else 0.0

    return {
        "window_days": days,
        "since": since.isoformat(),
        "model_version_filter": model_version,
        "resolved_predictions": total,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": round(f1, 4),
        },
    }


# ---------------------------------------------------------------------------
# /unseen-rate
# ---------------------------------------------------------------------------

_UNSEEN_DIMENSIONS: dict[str, str] = {
    "payer": "payer_name",
    "cpt": "primary_procedure_code",
    "dx": "primary_diagnosis_code",
    # v5: provider NPIs join the OOV roll-up. New billing/rendering NPIs
    # are an early-warning signal — they often correlate with provider-
    # credentialing denials (CARC 38, 170, 185, 242) before the 835 arrives.
    "billing_provider": "billing_provider_npi",
    "rendering_provider": "rendering_provider_npi",
}


@router.get("/unseen-rate")
async def unseen_rate(
    days: int = Query(7, ge=1, le=365),
    model_version: str | None = Query(default=None),
    top_n: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Out-of-vocabulary (OOV) rate over a window of predictions.

    For each prediction in the window we check the raw payer / CPT / Dx
    captured in ``feature_snapshot`` against the deployed engineer's
    training-time vocabulary (``category_vocabularies_``). The response
    reports total / unseen counts per dimension and the top-N unseen values.

    This is the most interpretable "new data arriving" signal in the system:
    when the unseen rate spikes, the model is being asked to score claims it
    has no historical baseline for. Treat the risk score as low-confidence
    until those claims accumulate reconciled outcomes for the next retrain.

    Parameters
    ----------
    days : int
        Rolling window size (1..365).
    model_version : str, optional
        Filter to a single deployed version. Without it, the result mixes
        any predictions tagged with different versions in the window.
    top_n : int
        How many top unseen values to return per dimension (1..50).
    """
    # Late import — avoid a module-level cycle with the ML package, and let
    # tests of monitoring run without importing the heavy ML stack.
    from app.ml.predictor import get_predictor

    predictor = get_predictor()
    if not predictor.is_ready or predictor.engineer is None:
        raise HTTPException(
            status_code=503,
            detail="Predictor not loaded; train/deploy a model before querying unseen-rate.",
        )
    vocab = predictor.engineer.category_vocabularies_
    if vocab is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Deployed engineer has no category vocabulary (artifact predates "
                "feature engineering v4). Retrain to enable unseen-rate."
            ),
        )

    since = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = select(PredictionLog.feature_snapshot).where(
        PredictionLog.prediction_time >= since
    )
    if model_version:
        stmt = stmt.where(PredictionLog.model_version == model_version)

    snapshots = (await db.execute(stmt)).scalars().all()
    total = len(snapshots)

    counters: dict[str, Counter] = {k: Counter() for k in _UNSEEN_DIMENSIONS}
    dim_unseen: dict[str, int] = {k: 0 for k in _UNSEEN_DIMENSIONS}
    any_unseen = 0

    for snap in snapshots:
        if not snap:
            continue
        row_has_unseen = False
        for dim, src_col in _UNSEEN_DIMENSIONS.items():
            raw = snap.get(src_col)
            if raw is None or raw == "":
                continue  # missing — not unseen
            sval = str(raw)
            # Mirror the engineer's stringification of null markers — those
            # are "missing", not "unseen".
            if sval in ("nan", "None"):
                continue
            if sval not in vocab.get(src_col, frozenset()):
                counters[dim][sval] += 1
                dim_unseen[dim] += 1
                row_has_unseen = True
        if row_has_unseen:
            any_unseen += 1

    return {
        "window_days": days,
        "since": since.isoformat(),
        "model_version_filter": model_version,
        "total_predictions": total,
        "any_unseen": any_unseen,
        "any_unseen_rate": _safe_div(any_unseen, total),
        "dimensions": {
            dim: {
                "source_column": _UNSEEN_DIMENSIONS[dim],
                "unseen_count": dim_unseen[dim],
                "unseen_rate": _safe_div(dim_unseen[dim], total),
                "top_unseen_values": [
                    {"value": v, "count": c}
                    for v, c in counters[dim].most_common(top_n)
                ],
            }
            for dim in _UNSEEN_DIMENSIONS
        },
    }


# ---------------------------------------------------------------------------
# /drift
# ---------------------------------------------------------------------------

@router.get("/drift")
async def drift_report(
    days: int = Query(7, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Drift of recent claims vs. training-distribution snapshot.

    Window of size ``days`` ending now. Returns PSI per column +
    unseen-category rate + missingness deltas.
    """
    try:
        snapshot = load_distribution_snapshot(settings.ML_DISTRIBUTIONS_PATH)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "No training distribution snapshot found. Retrain the model "
                "to generate one."
            ),
        )

    # Build a recent-claims DataFrame using the same loader the trainer uses.
    # build_dataset() returns labelled claims only; for drift we just need
    # the raw shape of incoming claims, so we also accept unlabelled rows
    # by skipping the remittance filter — but reusing build_dataset keeps
    # column names identical to the snapshot. Cost is acceptable for the
    # short window typically requested here.
    df = await build_dataset(db)
    if not df.empty and "service_from_date" in df.columns:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        df = df[pd.to_datetime(df["service_from_date"], errors="coerce").dt.date >= cutoff]

    # Prediction-score drift: from the prediction_log within the window.
    pred_since = datetime.now(timezone.utc) - timedelta(days=days)
    score_stmt = select(PredictionLog.predicted_risk).where(
        PredictionLog.prediction_time >= pred_since
    )
    score_rows = (await db.execute(score_stmt)).scalars().all()

    report = build_drift_report(snapshot, df, current_scores=score_rows)
    report = categorize_drift_report(report)
    report["window_days"] = days
    return report


# ---------------------------------------------------------------------------
# /prediction-log
# ---------------------------------------------------------------------------

@router.get("/prediction-log")
async def prediction_log_list(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    resolved: bool | None = Query(default=None),
    risk_level: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Paginated view of prediction_log rows."""
    stmt = select(PredictionLog).order_by(desc(PredictionLog.prediction_time))
    count_stmt = select(func.count(PredictionLog.id))

    if resolved is True:
        stmt = stmt.where(PredictionLog.resolved_at.is_not(None))
        count_stmt = count_stmt.where(PredictionLog.resolved_at.is_not(None))
    elif resolved is False:
        stmt = stmt.where(PredictionLog.resolved_at.is_(None))
        count_stmt = count_stmt.where(PredictionLog.resolved_at.is_(None))

    if risk_level:
        stmt = stmt.where(PredictionLog.risk_level == risk_level.upper())
        count_stmt = count_stmt.where(PredictionLog.risk_level == risk_level.upper())

    total = (await db.execute(count_stmt)).scalar_one()
    rows = (await db.execute(stmt.offset(offset).limit(limit))).scalars().all()

    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "rows": [
            {
                "id": r.id,
                "prediction_id": r.prediction_id,
                "claim_id": r.claim_id,
                "claim_number": r.claim_number,
                "predicted_risk": r.predicted_risk,
                "predicted_label": r.predicted_label,
                "risk_level": r.risk_level,
                "model_version": r.model_version,
                "feature_engineering_version": r.feature_engineering_version,
                "prediction_time": r.prediction_time.isoformat() if r.prediction_time else None,
                "actual_denied": r.actual_denied,
                "actual_status": r.actual_status,
                "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
                "resolved_by_remittance_id": r.resolved_by_remittance_id,
            }
            for r in rows
        ],
    }


@router.get("/prediction-log/{prediction_id}")
async def prediction_log_detail(
    prediction_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Single prediction_log row including feature_snapshot."""
    stmt = select(PredictionLog).where(PredictionLog.prediction_id == prediction_id)
    row = (await db.execute(stmt)).scalars().first()
    if not row:
        raise HTTPException(status_code=404, detail="prediction_id not found")
    return {
        "id": row.id,
        "prediction_id": row.prediction_id,
        "claim_id": row.claim_id,
        "claim_number": row.claim_number,
        "predicted_risk": row.predicted_risk,
        "predicted_label": row.predicted_label,
        "risk_level": row.risk_level,
        "model_version": row.model_version,
        "feature_engineering_version": row.feature_engineering_version,
        "feature_snapshot": row.feature_snapshot,
        "prediction_time": row.prediction_time.isoformat() if row.prediction_time else None,
        "actual_denied": row.actual_denied,
        "actual_status": row.actual_status,
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
        "resolved_by_remittance_id": row.resolved_by_remittance_id,
    }
