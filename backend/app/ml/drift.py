"""Lightweight drift calculations against a training-distribution snapshot.

All functions here are pure: they take a snapshot dict (from
``app.ml.distributions``) and a current DataFrame / array, and return
dicts of numbers. No DB, no IO. Callers decide what to persist or expose.

PSI interpretation (industry convention):
    < 0.10  : no significant change
    0.10 - 0.25 : moderate drift (investigate)
    >= 0.25 : significant drift (action recommended)
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

# Avoid log(0) in PSI by flooring small proportions.
_EPS = 1e-6

_OTHER_KEY = "__other__"
_MISSING_KEY = "__missing__"
_SEEN_KEY = "__seen_categories__"


def _normalize_cat_dist(
    reference: dict, current_counts: dict[str, int]
) -> tuple[dict[str, float], dict[str, float]]:
    """Align reference + current category distributions on the same key set.

    Treats any current category not in the reference as ``__other__``.
    """
    ref = {k: float(v) for k, v in reference.items() if not k.startswith("__seen_")}
    # Drop the seen-categories list when comparing distributions.
    seen = set(reference.get(_SEEN_KEY, []))

    cur_total = sum(current_counts.values())
    if cur_total == 0:
        cur = {k: 0.0 for k in ref}
        return ref, cur

    cur: dict[str, float] = {}
    other_mass = 0.0
    for value, count in current_counts.items():
        prop = float(count) / cur_total
        if value is None or value == "" or value == _MISSING_KEY:
            cur[_MISSING_KEY] = cur.get(_MISSING_KEY, 0.0) + prop
        elif value in ref or value in seen:
            cur[value] = cur.get(value, 0.0) + prop
        else:
            other_mass += prop
    if other_mass > 0 or _OTHER_KEY in ref:
        cur[_OTHER_KEY] = other_mass

    # Make sure both dicts share the same key universe.
    all_keys = set(ref) | set(cur)
    ref_aligned = {k: ref.get(k, 0.0) for k in all_keys}
    cur_aligned = {k: cur.get(k, 0.0) for k in all_keys}
    return ref_aligned, cur_aligned


def psi(reference: dict[str, float], current: dict[str, float]) -> float:
    """Population Stability Index between two aligned proportion dicts."""
    total = 0.0
    keys = set(reference) | set(current)
    for k in keys:
        r = max(float(reference.get(k, 0.0)), _EPS)
        c = max(float(current.get(k, 0.0)), _EPS)
        total += (c - r) * math.log(c / r)
    return round(total, 6)


def categorical_drift(
    reference: dict, current_series: pd.Series
) -> dict:
    """PSI + unseen-rate for a single categorical column."""
    if not reference:
        return {"psi": 0.0, "unseen_rate": 0.0, "n": int(current_series.shape[0])}

    seen = set(reference.get(_SEEN_KEY, []))
    cleaned = current_series.dropna().astype(str)
    n = int(current_series.shape[0])
    n_seen = 0
    n_unseen = 0
    counts: dict[str, int] = {}
    for v in cleaned:
        counts[v] = counts.get(v, 0) + 1
        if v in seen:
            n_seen += 1
        else:
            n_unseen += 1
    # Track missing separately so missing isn't lumped into unseen.
    n_missing = n - len(cleaned)
    if n_missing:
        counts[_MISSING_KEY] = n_missing

    ref_aligned, cur_aligned = _normalize_cat_dist(reference, counts)
    return {
        "psi": psi(ref_aligned, cur_aligned),
        "unseen_rate": round(n_unseen / n, 6) if n else 0.0,
        "missing_rate": round(n_missing / n, 6) if n else 0.0,
        "n": n,
    }


def continuous_drift(reference: dict, current_series: pd.Series) -> dict:
    """Percent-shift on quantiles + a PSI over coarse quantile bins."""
    s = pd.to_numeric(current_series, errors="coerce").dropna()
    n = int(s.shape[0])
    if n == 0 or not reference or reference.get("count", 0) == 0:
        return {"psi": 0.0, "mean_shift_pct": 0.0, "p50_shift_pct": 0.0, "n": n}

    ref_quantiles = [
        reference.get("min", 0.0),
        reference.get("p25", 0.0),
        reference.get("p50", 0.0),
        reference.get("p75", 0.0),
        reference.get("p95", 0.0),
        reference.get("max", 0.0),
    ]
    # Bin current values by reference quantile edges to get a PSI.
    # Reference distribution across these bins is by construction
    # [0.25, 0.25, 0.25, 0.20, 0.05] (min->p25, p25->p50, p50->p75,
    # p75->p95, p95->max).
    ref_bin_props = [0.25, 0.25, 0.25, 0.20, 0.05]
    # Deduplicate edges to avoid pandas.cut raising on equal bins.
    edges = sorted(set(ref_quantiles))
    if len(edges) < 2:
        return {"psi": 0.0, "mean_shift_pct": 0.0, "p50_shift_pct": 0.0, "n": n}
    # Extend bin edges to capture out-of-range values.
    edges = [-math.inf] + edges[1:-1] + [math.inf] if len(edges) > 2 else [-math.inf, math.inf]
    cur_counts, _ = np.histogram(s.values, bins=edges)
    cur_props = (cur_counts / cur_counts.sum()).tolist() if cur_counts.sum() else [0.0] * len(cur_counts)

    # Align reference proportions with current bin count (in case edges
    # collapsed because of duplicate quantiles).
    while len(ref_bin_props) > len(cur_props):
        ref_bin_props.pop()
    while len(ref_bin_props) < len(cur_props):
        ref_bin_props.append(0.0)

    ref_map = {f"b{i}": p for i, p in enumerate(ref_bin_props)}
    cur_map = {f"b{i}": p for i, p in enumerate(cur_props)}

    ref_mean = float(reference.get("mean", 0.0))
    ref_p50 = float(reference.get("p50", 0.0))
    cur_mean = float(s.mean())
    cur_p50 = float(s.quantile(0.5))

    def _pct_shift(ref: float, cur: float) -> float:
        if abs(ref) < _EPS:
            return 0.0
        return round((cur - ref) / abs(ref) * 100, 4)

    return {
        "psi": psi(ref_map, cur_map),
        "mean_shift_pct": _pct_shift(ref_mean, cur_mean),
        "p50_shift_pct": _pct_shift(ref_p50, cur_p50),
        "n": n,
    }


def missingness_drift(
    reference: dict[str, float], current_df: pd.DataFrame
) -> dict:
    """Per-column delta between reference missing rate and current rate."""
    if current_df.empty:
        return {}
    out: dict[str, dict] = {}
    n = len(current_df)
    for col, ref_rate in reference.items():
        if col not in current_df.columns:
            continue
        cur_rate = float(current_df[col].isna().mean())
        out[col] = {
            "reference_rate": round(float(ref_rate), 6),
            "current_rate": round(cur_rate, 6),
            "delta_pct_points": round((cur_rate - float(ref_rate)) * 100, 4),
            "n": n,
        }
    return out


def score_distribution_drift(
    reference_histogram: dict, current_scores: Iterable[float] | None
) -> dict:
    """Drift of the prediction-score distribution vs. the training snapshot."""
    if current_scores is None:
        return {"psi": 0.0, "n": 0}
    arr = np.asarray(list(current_scores), dtype=float)
    if arr.size == 0 or not reference_histogram.get("bins"):
        return {"psi": 0.0, "n": int(arr.size)}
    bins = reference_histogram["bins"]
    ref_props = reference_histogram.get("proportions") or []
    cur_counts, _ = np.histogram(np.clip(arr, 0.0, 1.0), bins=bins)
    total = int(cur_counts.sum())
    cur_props = (cur_counts / total).tolist() if total > 0 else [0.0] * len(cur_counts)
    while len(ref_props) < len(cur_props):
        ref_props.append(0.0)
    ref_map = {f"b{i}": p for i, p in enumerate(ref_props)}
    cur_map = {f"b{i}": p for i, p in enumerate(cur_props)}
    return {
        "psi": psi(ref_map, cur_map),
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "p50": round(float(np.median(arr)), 6),
    }


def classify_psi(value: float) -> str:
    """Map a PSI number to a coarse severity label."""
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"


# ---------------------------------------------------------------------------
# Drift categorization
#
# Real-world malformed-claim patterns (e.g. payer X submits without a
# diagnosis code) are *legitimate* structural drift signals — they predict
# denials. We must NOT suppress them.
#
# A sudden catastrophic null spike (current_rate >> training_rate by >= 30
# percentage points, or current >= 50% when training < 10%) usually means
# the parser or upstream feed broke. That belongs in Operational drift.
# ---------------------------------------------------------------------------

_CATASTROPHIC_DELTA_PP = 30.0
_CATASTROPHIC_CURRENT_RATE = 0.50
_CATASTROPHIC_TRAINING_RATE = 0.10


def _is_catastrophic_missingness(entry: dict) -> bool:
    delta_pp = float(entry.get("delta_pct_points", 0.0))
    cur = float(entry.get("current_rate", 0.0))
    ref = float(entry.get("reference_rate", 0.0))
    if delta_pp >= _CATASTROPHIC_DELTA_PP:
        return True
    if cur >= _CATASTROPHIC_CURRENT_RATE and ref < _CATASTROPHIC_TRAINING_RATE:
        return True
    return False


def _structural_severity_from_deltas(rows: dict[str, dict]) -> str:
    """Severity for structural drift based on max missingness delta (pp)."""
    if not rows:
        return "stable"
    max_delta = max(abs(float(r.get("delta_pct_points", 0.0))) for r in rows.values())
    if max_delta < 5.0:
        return "stable"
    if max_delta < 20.0:
        return "moderate"
    return "significant"


def _operational_severity(
    catastrophic_count: int, score_psi: float, data_psi_max: float
) -> str:
    """Severity for operational drift.

    - any catastrophic missingness spike → at least moderate.
    - catastrophic + high score drift OR very-high score drift while data is
      stable → significant (something off in the pipeline).
    """
    if catastrophic_count == 0 and score_psi < 0.10:
        return "stable"
    if catastrophic_count > 0 and (score_psi >= 0.25 or data_psi_max < 0.10):
        # Catastrophic null spike with no matching data drift → ops issue.
        return "significant"
    if score_psi >= 0.25 and data_psi_max < 0.10:
        return "significant"
    return "moderate"


def categorize_drift_report(report: dict) -> dict:
    """Split a flat drift report into Structural / Business / Operational.

    Pure function: takes the output of ``build_drift_report`` and adds three
    new top-level keys without removing any existing keys (so older clients
    still work).
    """
    missingness = report.get("missingness", {}) or {}
    categorical = report.get("categorical", {}) or {}
    continuous = report.get("continuous", {}) or {}
    score = report.get("score", {}) or {}

    # ----- Structural vs Operational split of missingness rows -----
    structural_rows: dict[str, dict] = {}
    catastrophic_rows: dict[str, dict] = {}
    for col, entry in missingness.items():
        if _is_catastrophic_missingness(entry):
            catastrophic_rows[col] = entry
        else:
            structural_rows[col] = entry

    structural_severity = _structural_severity_from_deltas(structural_rows)
    max_struct_delta = 0.0
    max_struct_col = None
    for col, e in structural_rows.items():
        d = abs(float(e.get("delta_pct_points", 0.0)))
        if d > max_struct_delta:
            max_struct_delta = d
            max_struct_col = col

    structural = {
        "severity": structural_severity,
        "max_delta_pct_points": round(max_struct_delta, 4),
        "max_delta_column": max_struct_col,
        "missingness": structural_rows,
        "notes": (
            "Malformed / incomplete-claim patterns at expected real-world rates."
            if structural_severity == "stable"
            else "Elevated missing-field rates — investigate payer / facility patterns."
        ),
    }

    # ----- Business drift: categorical + continuous PSI -----
    business_candidates: list[tuple[str, float]] = []
    for col, r in categorical.items():
        business_candidates.append((col, float(r.get("psi", 0.0))))
    for col, r in continuous.items():
        business_candidates.append((col, float(r.get("psi", 0.0))))

    if business_candidates:
        max_biz_col, max_biz_psi = max(business_candidates, key=lambda x: x[1])
        business_severity = classify_psi(max_biz_psi)
    else:
        max_biz_col, max_biz_psi = None, 0.0
        business_severity = "stable"

    business = {
        "severity": business_severity,
        "max_psi": round(max_biz_psi, 6),
        "max_psi_feature": max_biz_col,
        "categorical": categorical,
        "continuous": continuous,
        "notes": (
            "Payer / CPT / Dx / charge distributions stable."
            if business_severity == "stable"
            else f"Distribution shift detected in {max_biz_col} (PSI={max_biz_psi:.3f})."
        ),
    }

    # ----- Operational drift: catastrophic spikes + unexplained score drift -----
    score_psi = float(score.get("psi", 0.0))
    operational_severity = _operational_severity(
        catastrophic_count=len(catastrophic_rows),
        score_psi=score_psi,
        data_psi_max=max_biz_psi,
    )

    op_notes_parts: list[str] = []
    if catastrophic_rows:
        op_notes_parts.append(
            f"{len(catastrophic_rows)} column(s) show catastrophic null spike(s)"
            " — likely ingestion / parser issue."
        )
    if score_psi >= 0.25 and max_biz_psi < 0.10:
        op_notes_parts.append(
            "Score distribution shifted significantly while input data is stable"
            " — investigate model or feature-pipeline change."
        )
    if not op_notes_parts:
        op_notes_parts.append(
            "No catastrophic missingness or unexplained score drift detected."
        )

    operational = {
        "severity": operational_severity,
        "catastrophic_missingness": catastrophic_rows,
        "catastrophic_count": len(catastrophic_rows),
        "score_drift": score,
        "notes": " ".join(op_notes_parts),
    }

    # ----- Overall: worst of the three -----
    severity_rank = {"stable": 0, "moderate": 1, "significant": 2}
    categories = {
        "structural": structural_severity,
        "business": business_severity,
        "operational": operational_severity,
    }
    worst_cat = max(categories, key=lambda k: severity_rank[categories[k]])
    overall_severity = categories[worst_cat]

    out = dict(report)  # preserve original keys
    out["structural"] = structural
    out["business"] = business
    out["operational"] = operational
    out["overall"] = {
        "severity": overall_severity,
        "worst_category": worst_cat if overall_severity != "stable" else None,
        "categories": categories,
    }
    return out


def build_drift_report(
    snapshot: dict,
    current_df: pd.DataFrame,
    current_scores: Iterable[float] | None = None,
) -> dict:
    """End-to-end drift report against the training snapshot."""
    report: dict = {
        "snapshot_trained_at": snapshot.get("trained_at"),
        "n_current": int(len(current_df)),
        "categorical": {},
        "continuous": {},
        "missingness": missingness_drift(
            snapshot.get("missingness", {}), current_df
        ),
        "score": score_distribution_drift(
            snapshot.get("score_histogram", {}), current_scores
        ),
        "summary": {"max_psi": 0.0, "max_psi_feature": None, "severity": "stable"},
    }

    for col, ref in snapshot.get("categorical", {}).items():
        if col in current_df.columns:
            r = categorical_drift(ref, current_df[col])
            r["severity"] = classify_psi(r["psi"])
            report["categorical"][col] = r

    for col, ref in snapshot.get("continuous", {}).items():
        if col in current_df.columns:
            r = continuous_drift(ref, current_df[col])
            r["severity"] = classify_psi(r["psi"])
            report["continuous"][col] = r

    # Roll-up: which single feature looks worst?
    candidates: list[tuple[str, float]] = []
    for col, r in report["categorical"].items():
        candidates.append((col, r["psi"]))
    for col, r in report["continuous"].items():
        candidates.append((col, r["psi"]))
    if candidates:
        worst_col, worst_psi = max(candidates, key=lambda x: x[1])
        report["summary"] = {
            "max_psi": worst_psi,
            "max_psi_feature": worst_col,
            "severity": classify_psi(worst_psi),
        }
    return report
