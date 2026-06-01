"""Denial prediction pipeline with explainability.

Loads the trained XGBClassifier and fitted FeatureEngineer, accepts a raw
claim dict, and returns denial probability, risk level, and top contributing
features with human-readable names.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from app.core.config import settings
from app.ml.feature_engineering import FEATURE_COLUMNS, FeatureEngineer
from app.ml.model_loader import load_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Human-readable display names for internal feature columns
# ---------------------------------------------------------------------------

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "total_charge_amount": "Charge Amount",
    "line_count": "Line Count",
    "diagnosis_count": "Diagnosis Count",
    "total_units": "Service Units",
    "service_duration_days": "Service Duration",
    "has_modifier": "Modifier Present",
    "multiple_lines": "Multiple Service Lines",
    "many_diagnoses": "High Diagnosis Count",
    "high_charge_claim": "High Charge Amount",
    "service_month": "Service Month",
    "service_day_of_week": "Day of Week",
    "weekend_service": "Weekend Service",
    "payer_name_encoded": "Payer",
    "frequency_code_encoded": "Claim Frequency",
    "facility_type_code_encoded": "Facility Type",
    "primary_procedure_code_encoded": "Procedure Code",
    "primary_diagnosis_code_encoded": "Diagnosis Code",
    "place_of_service_encoded": "Place of Service",
    "total_billed_amount": "Billed Amount",
    "missing_payer": "Missing Payer",
    "missing_diagnosis": "Missing Diagnosis",
    "missing_procedure": "Missing Procedure",
    "missing_pos": "Missing Place of Service",
    "payer_cpt_denial_rate": "Payer × Procedure History",
    "payer_dx_denial_rate": "Payer × Diagnosis History",
    "payer_pos_denial_rate": "Payer × Place of Service History",
    "payer_volume": "Payer Claim Volume",
    "cpt_volume": "Procedure Claim Volume",
    "dx_volume": "Diagnosis Claim Volume",
    "is_rare_payer": "Rare Payer",
    "is_rare_cpt": "Rare Procedure",
    # v4 unseen-at-training indicators — distinct from "rare" (rare = seen
    # few times; unseen = never seen at all → no historical baseline).
    "unseen_payer": "New Payer (no training history)",
    "unseen_cpt": "New Procedure Code (no training history)",
    "unseen_dx": "New Diagnosis Code (no training history)",
    "unseen_any": "New Payer/CPT/Dx/Provider Combination",
    # v5 — authorization / referral / provider NPI / timely-filing signals.
    "has_prior_authorization": "Prior Authorization Present",
    "has_referral": "Referral Present",
    "billing_provider_npi_encoded": "Billing Provider",
    "rendering_provider_npi_encoded": "Rendering Provider",
    "unseen_billing_provider": "New Billing Provider (no training history)",
    "unseen_rendering_provider": "New Rendering Provider (no training history)",
    "service_to_submission_days": "Days from Service to Submission",
}

MODEL_VERSION = "v3.5"
FEATURE_VERSION = "v5"

# Fallback when no calibrator artifact is present (older models): use raw
# scores and the classic 0.5 cutoff so prediction never hard-fails.
DEFAULT_THRESHOLD = 0.5

# Risk-level bucket cut-offs operate on the CALIBRATED probability:
#   HIGH    >= decision threshold (predicted-denied)
#   MEDIUM  in [LOW_PROB_CUTOFF, threshold)   — borderline / worth reviewing
#   LOW     < LOW_PROB_CUTOFF                 — calibrated < 5% denial chance
# v3.4: LOW was previously 0.5 * threshold (~0.17), which made LOW span a band
# with ~1.7% real-world denial rate on the test set. Tightening LOW to <0.05
# drops the in-bucket denial rate to ~0.79%, matching the user expectation
# that "LOW = safe to ignore." MEDIUM widens to ~12% rate (review-worthy).
LOW_PROB_CUTOFF = 0.05


# ---------------------------------------------------------------------------
# DenialPredictor
# ---------------------------------------------------------------------------

class DenialPredictor:
    """Singleton-style predictor that loads model + feature engineer once."""

    def __init__(self) -> None:
        self.model: XGBClassifier | None = None
        self.engineer: FeatureEngineer | None = None
        self.calibrator = None
        self.threshold: float = DEFAULT_THRESHOLD
        self._loaded = False

    def load(self) -> bool:
        """Load model and feature engineer from disk. Returns True if successful."""
        self.model = load_model()
        if self.model is None:
            logger.warning("Model file not found at %s", settings.ML_MODEL_PATH)
            return False

        try:
            self.engineer = FeatureEngineer.load(settings.ML_ENCODERS_PATH)
        except Exception:
            logger.exception("Failed to load FeatureEngineer from %s", settings.ML_ENCODERS_PATH)
            self.model = None
            return False

        # Calibrator + tuned threshold are optional only for models trained
        # before calibration existed. When present, the bundled model_version
        # MUST match the predictor's MODEL_VERSION — a mismatch means the
        # calibrator was fit on a different model's score distribution and
        # would silently emit wrong probabilities.
        self.calibrator = None
        self.threshold = DEFAULT_THRESHOLD
        try:
            if os.path.exists(settings.ML_CALIBRATOR_PATH):
                bundle = joblib.load(settings.ML_CALIBRATOR_PATH)
                bundled_version = bundle.get("model_version")
                if bundled_version and bundled_version != MODEL_VERSION:
                    raise ValueError(
                        f"Calibrator model_version mismatch: bundled={bundled_version!r}, "
                        f"current={MODEL_VERSION!r}. Refusing to serve — retrain to align."
                    )
                self.calibrator = bundle.get("calibrator")
                self.threshold = float(bundle.get("threshold", DEFAULT_THRESHOLD))
                logger.info(
                    "Loaded calibrator (method=%s, model_version=%s, threshold=%.4f)",
                    bundle.get("method", "unknown"),
                    bundled_version or "unspecified",
                    self.threshold,
                )
            else:
                logger.warning(
                    "No calibrator at %s; serving raw scores at 0.5 threshold",
                    settings.ML_CALIBRATOR_PATH,
                )
        except ValueError:
            # Re-raise version-mismatch errors loudly — silent fallback would
            # mask a deployment bug.
            self.model = None
            raise
        except Exception:
            logger.exception("Failed to load calibrator; falling back to raw scores")
            self.calibrator = None
            self.threshold = DEFAULT_THRESHOLD

        self._loaded = True
        logger.info("DenialPredictor loaded successfully")
        return True

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.model is not None and self.engineer is not None

    def predict(self, claim: dict) -> dict:
        """Run prediction on a single claim.

        Thin wrapper over ``predict_batch``: builds a 1-element batch and
        returns the single result. The batched code path is the source of
        truth — any change to scoring/explainability logic only needs to
        land once. Output is byte-identical to the previous single-claim
        implementation (the verification harness in
        ``scripts/verify_batched_predict.py`` asserts this).

        Parameters
        ----------
        claim : dict
            Raw claim fields matching the FeatureEngineer input schema.

        Returns
        -------
        dict
            prediction_id, risk_score, risk_level, top_risk_factors, metadata.
        """
        return self.predict_batch([claim])[0]

    def predict_batch(self, claims: list[dict]) -> list[dict]:
        """Vectorized prediction over a list of raw claim dicts.

        Runs ``FeatureEngineer.transform``, ``model.predict_proba``,
        ``calibrator.transform``, and tree-SHAP ``pred_contribs`` ONCE for
        the whole batch, then splits the per-row results. Eliminates the
        ~40 ms/claim pandas fixed cost that dominates single-claim
        latency, yielding ~60× speedup on ``/predict-file`` for a
        1000-claim batch.

        Each per-row result is byte-identical to what ``predict()`` would
        have produced for that claim alone (same encoder state, same
        booster, same calibrator, and SHAP contributions are per-row by
        construction — they do not couple across the batch).
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded. Call load() first.")
        if not claims:
            return []

        df = pd.DataFrame(claims)
        features = self.engineer.transform(df)

        # Hard-fail validation: feature count must match what the engineer
        # was fit with. Silent column drift here would propagate into
        # XGBoost as a shape error or, worse, a silent miscolumn mapping.
        expected = self.engineer.n_features_in_
        if expected is not None and features.shape[1] != expected:
            raise RuntimeError(
                f"Feature matrix has {features.shape[1]} columns, "
                f"engineer expected {expected}. The engineer/model artifact is "
                "inconsistent — retrain to regenerate."
            )

        # Vectorized scoring across the whole batch.
        raw_scores = self.model.predict_proba(features)[:, 1]  # shape (N,)

        # Calibrate to true probabilities. Isotonic is monotonic +
        # vectorized; per-row output matches a per-row call exactly.
        if self.calibrator is not None:
            cal_scores = self.calibrator.transform(raw_scores)
            risk_scores = np.clip(cal_scores, 0.0, 1.0)
        else:
            risk_scores = raw_scores

        # Per-row tree-SHAP attributions in ONE booster call. Tree-SHAP
        # decomposes each row independently (no batch coupling), so per-row
        # contributions equal what predict() computed one row at a time.
        contribs_batch = self._compute_contribs_batch(features)

        unseen_columns = self._unseen_columns_present(features)

        # Single timestamp for the batch — matches the wall-clock semantics
        # of "this file was scored at T"; predict() also calls now() once
        # per call, so 1-row batches keep their single-timestamp behaviour.
        now_iso = datetime.now(timezone.utc).isoformat()

        results: list[dict] = []
        for i in range(len(claims)):
            risk_score = float(risk_scores[i])
            raw_score = float(raw_scores[i])

            top_risk_factors = self._format_contribution_row(
                contribs_batch[i, :-1]  # drop bias term
            )

            unseen_indicators = (
                self._build_unseen_indicators(features, i, unseen_columns)
                if unseen_columns is not None
                else None
            )

            results.append({
                "prediction_id": str(uuid.uuid4()),
                "risk_score": round(risk_score, 4),
                "raw_risk_score": round(raw_score, 4),
                "predicted_label": int(risk_score >= self.threshold),
                "decision_threshold": round(self.threshold, 4),
                "risk_level": _classify_risk(risk_score, self.threshold),
                "top_risk_factors": top_risk_factors,
                "unseen_indicators": unseen_indicators,
                "model_version": MODEL_VERSION,
                "feature_version": FEATURE_VERSION,
                "prediction_timestamp": now_iso,
            })

        return results

    # ---- private: shared per-row formatting ------------------------------

    _UNSEEN_FEATURES: tuple[str, ...] = (
        "unseen_payer",
        "unseen_cpt",
        "unseen_dx",
        "unseen_billing_provider",
        "unseen_rendering_provider",
        "unseen_any",
    )

    def _unseen_columns_present(self, features: pd.DataFrame) -> tuple[str, ...] | None:
        """Return the tuple of unseen-* feature columns iff all are present."""
        if all(c in features.columns for c in self._UNSEEN_FEATURES):
            return self._UNSEEN_FEATURES
        return None

    @staticmethod
    def _build_unseen_indicators(
        features: pd.DataFrame, row_idx: int, cols: tuple[str, ...]
    ) -> dict:
        """Per-row unseen-indicators dict, matching predict()'s shape exactly."""
        row = features.iloc[row_idx]
        return {
            "payer": bool(int(row["unseen_payer"])),
            "cpt": bool(int(row["unseen_cpt"])),
            "dx": bool(int(row["unseen_dx"])),
            "billing_provider": bool(int(row["unseen_billing_provider"])),
            "rendering_provider": bool(int(row["unseen_rendering_provider"])),
            "any": bool(int(row["unseen_any"])),
        }

    def _compute_contribs_batch(self, features: pd.DataFrame) -> np.ndarray:
        """One ``booster.predict(pred_contribs=True)`` call for the batch.

        Shape ``(N, n_features + 1)`` — last column is the bias term. Caller
        slices each row and passes it to ``_format_contribution_row``.
        """
        booster = self.model.get_booster()
        import xgboost as xgb

        dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
        return booster.predict(dmatrix, pred_contribs=True)

    @staticmethod
    def _format_contribution_row(contrib_values: np.ndarray) -> list[dict]:
        """Convert a row of raw SHAP values into the top-5 display list.

        Identical formatting to the previous ``_compute_contributions`` path
        (impact string rounding, direction labels, top-5 sort by absolute
        contribution). Pure function — does not touch self — so callers can
        slice a batched contributions matrix and call this per row without
        any cross-row coupling.
        """
        # Normalize: express each contribution as a percentage of total
        # absolute contribution (the denial risk signal magnitude).
        abs_sum = float(np.abs(contrib_values).sum())
        if abs_sum == 0:
            abs_sum = 1.0  # avoid division by zero

        # Build list of (feature_name, contribution, normalized_pct)
        feature_contribs = []
        for i, col in enumerate(FEATURE_COLUMNS):
            val = float(contrib_values[i])
            pct = (val / abs_sum) * 100
            display_name = FEATURE_DISPLAY_NAMES.get(col, col)
            direction = "risk" if val >= 0 else "protective"
            feature_contribs.append({
                "feature": display_name,
                "impact": f"{'+' if val >= 0 else ''}{pct:.0f}%",
                "direction": direction,
                "_abs": abs(val),
            })

        # Sort by absolute contribution, take top 5
        feature_contribs.sort(key=lambda x: x["_abs"], reverse=True)
        top_5 = feature_contribs[:5]

        # Remove internal sort key
        for item in top_5:
            del item["_abs"]

        return top_5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_risk(score: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Bucket a calibrated probability into LOW / MEDIUM / HIGH.

    Buckets use a fixed LOW boundary (calibrated probability) so "LOW" carries
    a consistent operational meaning across model versions, while HIGH stays
    coupled to the model's tuned decision threshold:
      - HIGH   : score >= threshold (predicted denied)
      - MEDIUM : LOW_PROB_CUTOFF <= score < threshold (borderline / review)
      - LOW    : score < LOW_PROB_CUTOFF (very-low denial probability)
    v3.4: LOW boundary detached from threshold and set at a fixed 5%. On the
    test split this drops in-bucket denial rate from ~1.7% to ~0.8% while
    keeping HIGH semantics ("predicted denied") tied to the deployed
    threshold.
    """
    if score >= threshold:
        return "HIGH"
    if score >= LOW_PROB_CUTOFF:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_predictor: DenialPredictor | None = None


def get_predictor() -> DenialPredictor:
    """Return a lazily-initialized singleton DenialPredictor.

    If the very first load raises (e.g. strict-fail calibrator version
    mismatch), reset the module-level singleton to ``None`` before
    re-raising. Otherwise the broken instance would stick around and every
    subsequent caller would either see ``is_ready=False`` *or* hit the
    raised exception again with no way to recover short of restarting
    uvicorn — the training endpoint can't even heal it, because
    ``get_predictor()`` would just return the half-initialized object
    without re-running ``load``.
    """
    global _predictor
    if _predictor is None:
        try:
            instance = DenialPredictor()
            instance.load()
        except Exception:
            _predictor = None  # explicit — don't cache a broken instance
            raise
        _predictor = instance
    return _predictor


def reset_predictor() -> None:
    """Drop the cached singleton so the next ``get_predictor()`` call
    re-runs ``load``. Used by tests and by recovery flows after the
    on-disk artifacts have been replaced.
    """
    global _predictor
    _predictor = None
