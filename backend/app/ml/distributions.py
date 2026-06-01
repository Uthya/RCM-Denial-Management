"""Snapshot training-data distributions for downstream drift monitoring.

Writes a small JSON artifact at training time that captures:
  * top-N category frequencies for payer / CPT / Dx / POS / facility / freq
  * continuous quantiles for charge / units / line_count / diagnosis_count
  * missingness rates per source column
  * rare-flag rates produced by the feature engineer
  * a 10-bin histogram of training-set prediction scores (caller-supplied)

The artifact is deliberately compact and human-readable. It is consumed
by ``app.ml.drift`` (read-only) to compute PSI, unseen-rate, and
missingness deltas against incoming claims.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DISTRIBUTIONS_VERSION = "v2"
# v1 -> v2: provider NPI categoricals and service-to-submission day quantiles
# added to the snapshot so the drift monitor sees them. Backward-compatible:
# older snapshots without these keys simply get an empty drift entry, not
# an exception.

# Top-N values to keep per categorical column. The long tail is summarised
# in a single "__other__" bucket so PSI calculations stay stable when new
# rare categories appear.
_TOP_N_PAYER = 50
_TOP_N_CPT = 100
_TOP_N_DX = 100
_TOP_N_POS = 20
_TOP_N_FACILITY = 20
_TOP_N_FREQ = 10
# Providers are usually a 10-200 distinct-NPI tail per practice; keep 200
# to retain the heavy hitters and fold the rest into __other__.
_TOP_N_PROVIDER = 200

_OTHER_KEY = "__other__"
_MISSING_KEY = "__missing__"


def _category_frequencies(series: pd.Series, top_n: int) -> dict:
    """Return {value: proportion} with long tail folded into __other__."""
    if series.empty:
        return {}
    filled = series.astype(object).where(series.notna(), other=None)
    total = len(filled)
    counts = filled.value_counts(dropna=False).head(top_n)

    out: dict[str, float] = {}
    kept = 0
    for value, count in counts.items():
        key = _MISSING_KEY if value is None else str(value)
        out[key] = round(float(count) / total, 6)
        kept += int(count)

    remaining = total - kept
    if remaining > 0:
        out[_OTHER_KEY] = round(float(remaining) / total, 6)
    # Track the set of seen-during-training values for unseen-rate checks.
    out["__seen_categories__"] = sorted(
        [str(v) for v in series.dropna().unique().tolist()]
    )
    return out


def _continuous_summary(series: pd.Series) -> dict:
    """Return quantiles + mean for a numeric column."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0}
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std() if s.shape[0] > 1 else 0.0),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def _missingness_rates(df: pd.DataFrame, columns: list[str]) -> dict:
    """Return {column: fraction missing} for each named column."""
    total = len(df)
    if total == 0:
        return {col: 0.0 for col in columns}
    return {
        col: round(float(df[col].isna().mean()), 6)
        for col in columns
        if col in df.columns
    }


def _score_histogram(scores: np.ndarray) -> dict:
    """10-bin [0,1] histogram of predicted risk scores."""
    if scores is None or len(scores) == 0:
        return {"bins": [], "counts": []}
    edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(np.clip(scores, 0.0, 1.0), bins=edges)
    total = int(counts.sum())
    return {
        "bins": [round(float(e), 2) for e in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
        "proportions": [
            round(float(c) / total, 6) if total > 0 else 0.0 for c in counts.tolist()
        ],
        "n": total,
    }


def build_distribution_snapshot(
    raw_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    training_scores: np.ndarray | None,
    *,
    trained_at: str,
    feature_version: str,
    model_version: str,
) -> dict:
    """Produce the in-memory distribution snapshot dict.

    Pure function so unit tests can exercise it without touching disk.
    """
    snapshot = {
        "version": DISTRIBUTIONS_VERSION,
        "trained_at": trained_at,
        "feature_version": feature_version,
        "model_version": model_version,
        "n_rows": int(len(raw_df)),
        "categorical": {
            "payer_name": _category_frequencies(
                raw_df.get("payer_name", pd.Series(dtype=object)), _TOP_N_PAYER
            ),
            "primary_procedure_code": _category_frequencies(
                raw_df.get("primary_procedure_code", pd.Series(dtype=object)),
                _TOP_N_CPT,
            ),
            "primary_diagnosis_code": _category_frequencies(
                raw_df.get("primary_diagnosis_code", pd.Series(dtype=object)),
                _TOP_N_DX,
            ),
            "place_of_service": _category_frequencies(
                raw_df.get("place_of_service", pd.Series(dtype=object)),
                _TOP_N_POS,
            ),
            "facility_type_code": _category_frequencies(
                raw_df.get("facility_type_code", pd.Series(dtype=object)),
                _TOP_N_FACILITY,
            ),
            "frequency_code": _category_frequencies(
                raw_df.get("frequency_code", pd.Series(dtype=object)),
                _TOP_N_FREQ,
            ),
            # v5: provider NPIs go through the same PSI/unseen-rate machinery
            # as payer / CPT / Dx.
            "billing_provider_npi": _category_frequencies(
                raw_df.get("billing_provider_npi", pd.Series(dtype=object)),
                _TOP_N_PROVIDER,
            ),
            "rendering_provider_npi": _category_frequencies(
                raw_df.get("rendering_provider_npi", pd.Series(dtype=object)),
                _TOP_N_PROVIDER,
            ),
        },
        "continuous": {
            "total_charge_amount": _continuous_summary(
                raw_df.get("total_charge_amount", pd.Series(dtype=float))
            ),
            "total_billed_amount": _continuous_summary(
                raw_df.get("total_billed_amount", pd.Series(dtype=float))
            ),
            "total_units": _continuous_summary(
                raw_df.get("total_units", pd.Series(dtype=float))
            ),
            "line_count": _continuous_summary(
                raw_df.get("line_count", pd.Series(dtype=float))
            ),
            "diagnosis_count": _continuous_summary(
                raw_df.get("diagnosis_count", pd.Series(dtype=float))
            ),
            # v5: timely-filing distribution from the engineered feature.
            "service_to_submission_days": _continuous_summary(
                feature_df.get(
                    "service_to_submission_days", pd.Series(dtype=float)
                )
            ),
        },
        "missingness": _missingness_rates(
            raw_df,
            [
                "payer_name",
                "primary_procedure_code",
                "primary_diagnosis_code",
                "place_of_service",
                "facility_type_code",
                # v5: missingness here is itself signal (no auth / referral
                # / NPI on a real claim correlates with denials), so it goes
                # in structural-drift not operational-drift.
                "authorization_number",
                "referral_number",
                "billing_provider_npi",
                "rendering_provider_npi",
            ],
        ),
        "rare_flags": {
            "is_rare_payer_rate": (
                round(float(feature_df["is_rare_payer"].mean()), 6)
                if "is_rare_payer" in feature_df.columns
                else 0.0
            ),
            "is_rare_cpt_rate": (
                round(float(feature_df["is_rare_cpt"].mean()), 6)
                if "is_rare_cpt" in feature_df.columns
                else 0.0
            ),
        },
        "score_histogram": _score_histogram(training_scores),
    }
    return snapshot


def save_distribution_snapshot(snapshot: dict, path: str | Path) -> None:
    """Persist the snapshot to disk as pretty-printed JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snapshot, indent=2, default=str))
    logger.info("Saved training distribution snapshot to %s", p)


def load_distribution_snapshot(path: str | Path) -> dict:
    """Read a saved snapshot. Raises FileNotFoundError if missing."""
    return json.loads(Path(path).read_text())
