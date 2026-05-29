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
    "unseen_any": "New Payer/CPT/Dx Combination",
}

MODEL_VERSION = "v3.4"
FEATURE_VERSION = "v4"

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

        Parameters
        ----------
        claim : dict
            Raw claim fields matching the FeatureEngineer input schema.

        Returns
        -------
        dict
            prediction_id, risk_score, risk_level, top_risk_factors, metadata.
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build a 1-row DataFrame from the claim dict
        df = pd.DataFrame([claim])

        # Transform through the feature pipeline
        features = self.engineer.transform(df)

        # Hard-fail validation: feature count must match what the engineer
        # was fit with. Silent column drift here would propagate into XGBoost
        # as a shape error or, worse, a silent miscolumn mapping.
        expected = self.engineer.n_features_in_
        if expected is not None and features.shape[1] != expected:
            raise RuntimeError(
                f"Feature matrix has {features.shape[1]} columns, "
                f"engineer expected {expected}. The engineer/model artifact is "
                "inconsistent — retrain to regenerate."
            )

        # Raw model probability of denial (class 1)
        proba = self.model.predict_proba(features)
        raw_score = float(proba[0, 1])

        # Calibrate to a true probability (isotonic). The risk_score we expose
        # is the calibrated value so "0.7" means ~70% denial likelihood, which
        # is what risk_level and the decision threshold are defined against.
        if self.calibrator is not None:
            cal = float(self.calibrator.transform([raw_score])[0])
            risk_score = min(max(cal, 0.0), 1.0)
        else:
            risk_score = raw_score

        # Risk level (on the calibrated score), tied to the decision threshold:
        # HIGH == would be predicted denied, MEDIUM == borderline, LOW == clear.
        risk_level = _classify_risk(risk_score, self.threshold)

        # Binary decision at the tuned threshold (not a hardcoded 0.5).
        predicted_label = int(risk_score >= self.threshold)

        # Feature contributions via XGBoost booster pred_contribs. Computed on
        # the raw model (calibration is a monotonic post-transform and does not
        # change feature attributions), so explainability is unaffected.
        top_risk_factors = self._compute_contributions(features)

        # Unseen-category indicators (v4 features). Surfacing these lets the UI
        # badge claims whose payer/CPT/Dx wasn't in the training vocabulary —
        # the user sees "the model has no historical baseline for this value"
        # instead of mistakenly trusting a low score on an OOV claim. Pulled
        # directly from the engineered row so the badge state always matches
        # what the model itself saw.
        unseen_indicators: dict | None = None
        if all(c in features.columns for c in ("unseen_payer", "unseen_cpt", "unseen_dx", "unseen_any")):
            unseen_indicators = {
                "payer": bool(int(features["unseen_payer"].iloc[0])),
                "cpt": bool(int(features["unseen_cpt"].iloc[0])),
                "dx": bool(int(features["unseen_dx"].iloc[0])),
                "any": bool(int(features["unseen_any"].iloc[0])),
            }

        return {
            "prediction_id": str(uuid.uuid4()),
            "risk_score": round(risk_score, 4),
            "raw_risk_score": round(raw_score, 4),
            "predicted_label": predicted_label,
            "decision_threshold": round(self.threshold, 4),
            "risk_level": risk_level,
            "top_risk_factors": top_risk_factors,
            "unseen_indicators": unseen_indicators,
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _compute_contributions(self, features: pd.DataFrame) -> list[dict]:
        """Compute per-feature contributions using XGBoost's built-in tree SHAP.

        Returns the top 5 features sorted by absolute contribution, with
        human-readable names and normalized impact percentages.
        """
        booster = self.model.get_booster()
        import xgboost as xgb

        dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
        contribs = booster.predict(dmatrix, pred_contribs=True)

        # contribs shape: (1, n_features + 1) — last column is the bias term
        contrib_values = contribs[0, :-1]  # exclude bias

        # Normalize: express each contribution as a percentage of total positive
        # contribution sum (the denial risk signal)
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
    """Return a lazily-initialized singleton DenialPredictor."""
    global _predictor
    if _predictor is None:
        _predictor = DenialPredictor()
        _predictor.load()
    return _predictor
