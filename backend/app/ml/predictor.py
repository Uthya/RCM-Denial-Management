"""Denial prediction pipeline with explainability.

Loads the trained XGBClassifier and fitted FeatureEngineer, accepts a raw
claim dict, and returns denial probability, risk level, and top contributing
features with human-readable names.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

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
}

MODEL_VERSION = "v1"
FEATURE_VERSION = "v2"


# ---------------------------------------------------------------------------
# DenialPredictor
# ---------------------------------------------------------------------------

class DenialPredictor:
    """Singleton-style predictor that loads model + feature engineer once."""

    def __init__(self) -> None:
        self.model: XGBClassifier | None = None
        self.engineer: FeatureEngineer | None = None
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

        # Predict probability of denial (class 1)
        proba = self.model.predict_proba(features)
        risk_score = float(proba[0, 1])

        # Risk level
        risk_level = _classify_risk(risk_score)

        # Feature contributions via XGBoost booster pred_contribs
        top_risk_factors = self._compute_contributions(features)

        return {
            "prediction_id": str(uuid.uuid4()),
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "top_risk_factors": top_risk_factors,
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

def _classify_risk(score: float) -> str:
    if score < 0.3:
        return "LOW"
    elif score <= 0.7:
        return "MEDIUM"
    else:
        return "HIGH"


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
