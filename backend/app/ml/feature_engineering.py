"""Feature engineering for claim-level denial prediction.

Transforms the raw DataFrame from ``dataset.build_dataset()`` into a
23-feature numeric matrix suitable for XGBoost training and inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

from app.core.config import settings

logger = logging.getLogger(__name__)

FEATURE_ENGINEERING_VERSION = "v2"

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_LEAKAGE_COLUMNS: frozenset[str] = frozenset({
    "denied",
    "claim_id",
    "claim_number",
    # Any remittance / adjustment / CARC data that would leak the outcome
})

_CATEGORICAL_COLUMNS: list[str] = [
    "payer_name",
    "frequency_code",
    "facility_type_code",
    "primary_procedure_code",
    "primary_diagnosis_code",
    "place_of_service",
]

_MISSINGNESS_SOURCES: dict[str, str] = {
    "missing_payer": "payer_name",
    "missing_diagnosis": "primary_diagnosis_code",
    "missing_procedure": "primary_procedure_code",
    "missing_pos": "place_of_service",
}

FEATURE_COLUMNS: list[str] = [
    "total_charge_amount",
    "line_count",
    "diagnosis_count",
    "total_units",
    "service_duration_days",
    "has_modifier",
    "multiple_lines",
    "many_diagnoses",
    "high_charge_claim",
    "service_month",
    "service_day_of_week",
    "weekend_service",
    "payer_name_encoded",
    "frequency_code_encoded",
    "facility_type_code_encoded",
    "primary_procedure_code_encoded",
    "primary_diagnosis_code_encoded",
    "place_of_service_encoded",
    "total_billed_amount",
    "missing_payer",
    "missing_diagnosis",
    "missing_procedure",
    "missing_pos",
]


# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureMeta:
    """Describes one output feature."""

    name: str
    dtype: str
    source_column: str
    transformation: str


_FEATURE_METADATA: list[FeatureMeta] = [
    FeatureMeta("total_charge_amount", "float64", "total_charge_amount", "passthrough"),
    FeatureMeta("line_count", "int64", "line_count", "passthrough"),
    FeatureMeta("diagnosis_count", "int64", "diagnosis_count", "passthrough"),
    FeatureMeta("total_units", "float64", "total_units", "passthrough"),
    FeatureMeta("service_duration_days", "int64", "service_from_date / service_to_date", "date diff (days), 0 if null"),
    FeatureMeta("has_modifier", "int64", "has_modifier", "cast to int"),
    FeatureMeta("multiple_lines", "int64", "line_count", "> 1"),
    FeatureMeta("many_diagnoses", "int64", "diagnosis_count", "> 5"),
    FeatureMeta("high_charge_claim", "int64", "total_charge_amount", "> 75th percentile (fitted)"),
    FeatureMeta("service_month", "int64", "service_from_date", "month 1-12"),
    FeatureMeta("service_day_of_week", "int64", "service_from_date", "0=Mon..6=Sun"),
    FeatureMeta("weekend_service", "int64", "service_from_date", "day_of_week >= 5"),
    FeatureMeta("payer_name_encoded", "float64", "payer_name", "TargetEncoder"),
    FeatureMeta("frequency_code_encoded", "float64", "frequency_code", "TargetEncoder"),
    FeatureMeta("facility_type_code_encoded", "float64", "facility_type_code", "TargetEncoder"),
    FeatureMeta("primary_procedure_code_encoded", "float64", "primary_procedure_code", "TargetEncoder"),
    FeatureMeta("primary_diagnosis_code_encoded", "float64", "primary_diagnosis_code", "TargetEncoder"),
    FeatureMeta("place_of_service_encoded", "float64", "place_of_service", "TargetEncoder"),
    FeatureMeta("total_billed_amount", "float64", "total_billed_amount", "passthrough"),
    FeatureMeta("missing_payer", "int64", "payer_name", "is null"),
    FeatureMeta("missing_diagnosis", "int64", "primary_diagnosis_code", "is null"),
    FeatureMeta("missing_procedure", "int64", "primary_procedure_code", "is null"),
    FeatureMeta("missing_pos", "int64", "place_of_service", "is null"),
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_service_duration(
    from_dates: pd.Series,
    to_dates: pd.Series,
) -> pd.Series:
    """Return integer days between service dates.

    - Null dates -> 0
    - Same-day -> 0
    - Negative durations clipped to 0
    """
    from_dt = pd.to_datetime(from_dates, errors="coerce")
    to_dt = pd.to_datetime(to_dates, errors="coerce")
    delta = (to_dt - from_dt).dt.days
    return delta.fillna(0).clip(lower=0).astype("int64")


def _validate_features(df: pd.DataFrame) -> None:
    """Assert the feature matrix is well-formed.

    Raises ``ValueError`` with a clear message identifying which columns fail.
    """
    if len(df.columns) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} feature columns, got {len(df.columns)}: {list(df.columns)}"
        )

    # No nulls
    null_cols = [c for c in df.columns if df[c].isna().any()]
    if null_cols:
        raise ValueError(f"Null values found in columns: {null_cols}")

    # No infs
    numeric_df = df.select_dtypes(include=[np.number])
    inf_cols = [c for c in numeric_df.columns if np.isinf(numeric_df[c]).any()]
    if inf_cols:
        raise ValueError(f"Inf values found in columns: {inf_cols}")

    # No object dtypes
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        raise ValueError(f"Object dtype columns found: {obj_cols}")


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """Transforms a raw claim DataFrame into a 23-feature numeric matrix.

    Follows the scikit-learn fit/transform convention.
    """

    _BOOL_INT_FEATURES: ClassVar[list[str]] = [
        "has_modifier",
        "multiple_lines",
        "many_diagnoses",
        "high_charge_claim",
        "weekend_service",
        "missing_payer",
        "missing_diagnosis",
        "missing_procedure",
        "missing_pos",
        "service_month",
        "service_day_of_week",
        "service_duration_days",
        "line_count",
        "diagnosis_count",
    ]

    def __init__(self) -> None:
        self.target_encoder_: TargetEncoder | None = None
        self.high_charge_threshold_: float | None = None
        self._is_fitted: bool = False

    # ---- private helpers -------------------------------------------------

    def _build_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all non-categorical features.

        Shared by ``transform()`` and ``fit_transform()`` to avoid
        code duplication.
        """
        out = pd.DataFrame(index=df.index)

        # 1-4: Numeric passthrough
        out["total_charge_amount"] = df["total_charge_amount"].astype("float64")
        out["line_count"] = df["line_count"].astype("int64")
        out["diagnosis_count"] = df["diagnosis_count"].astype("int64")
        out["total_units"] = df["total_units"].astype("float64")

        # 5: Service duration
        out["service_duration_days"] = _compute_service_duration(
            df["service_from_date"], df["service_to_date"]
        )

        # 6: has_modifier -> int
        out["has_modifier"] = df["has_modifier"].astype(int)

        # 7: multiple_lines
        out["multiple_lines"] = (df["line_count"] > 1).astype(int)

        # 8: many_diagnoses
        out["many_diagnoses"] = (df["diagnosis_count"] > 5).astype(int)

        # 9: high_charge_claim (fitted threshold)
        out["high_charge_claim"] = (
            df["total_charge_amount"] > self.high_charge_threshold_
        ).astype(int)

        # 10-12: Date features from service_from_date
        from_dt = pd.to_datetime(df["service_from_date"], errors="coerce")
        out["service_month"] = from_dt.dt.month.fillna(1).astype("int64")
        out["service_day_of_week"] = from_dt.dt.dayofweek.fillna(0).astype("int64")
        out["weekend_service"] = (out["service_day_of_week"] >= 5).astype(int)

        return out

    def _finalize(self, out: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add remaining features, enforce column order, coerce types, validate.

        Called after categorical encoding columns have been added to ``out``.
        """
        # 19: total_billed_amount passthrough
        out["total_billed_amount"] = df["total_billed_amount"].astype("float64")

        # 20-23: Missingness indicators
        for feat_name, src_col in _MISSINGNESS_SOURCES.items():
            out[feat_name] = df[src_col].isna().astype(int)

        # Enforce column order
        out = out[FEATURE_COLUMNS]

        # Coerce types: bool/int columns -> int64, encoded columns -> float64
        for col in out.columns:
            if col in self._BOOL_INT_FEATURES:
                out[col] = out[col].astype("int64")
            elif col.endswith("_encoded"):
                out[col] = out[col].astype("float64")
            else:
                out[col] = out[col].astype("float64")

        # Validate
        _validate_features(out)

        logger.info("Feature matrix: %d rows x %d cols", len(out), len(out.columns))
        return out

    def _prepare_categorical_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract categorical columns as a string DataFrame for TargetEncoder."""
        cat_df = df[_CATEGORICAL_COLUMNS].copy()
        for col in _CATEGORICAL_COLUMNS:
            cat_df[col] = cat_df[col].astype(str).replace("nan", np.nan).replace("None", np.nan)
        return cat_df

    # ---- fit / transform / fit_transform --------------------------------

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        """Fit encoders and thresholds from the training DataFrame.

        Requires a ``denied`` column (the binary target) for TargetEncoder.
        """
        if "denied" not in df.columns:
            raise ValueError(
                "Training DataFrame must contain a 'denied' column "
                "(binary target) for TargetEncoder fitting."
            )

        y = df["denied"].astype(int)

        # TargetEncoder — fit on all 6 categorical columns at once
        cat_df = self._prepare_categorical_input(df)
        self.target_encoder_ = TargetEncoder(
            target_type="binary",
            smooth="auto",
            cv=5,
            random_state=42,
        )
        self.target_encoder_.fit(cat_df, y)

        # 75th percentile threshold for high_charge_claim
        self.high_charge_threshold_ = float(
            df["total_charge_amount"].quantile(0.75)
        )

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Produce the 23-column feature matrix from a raw DataFrame."""
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer has not been fitted. Call fit() first.")

        out = self._build_base_features(df)

        # 13-18: Categorical target encoding
        cat_df = self._prepare_categorical_input(df)
        encoded = self.target_encoder_.transform(cat_df)
        encoded_df = pd.DataFrame(
            encoded,
            columns=[f"{col}_encoded" for col in _CATEGORICAL_COLUMNS],
            index=df.index,
        )
        for col in encoded_df.columns:
            out[col] = encoded_df[col]

        return self._finalize(out, df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Uses TargetEncoder.fit_transform() which applies internal cross-fitting
        (5-fold CV) so that each training sample is encoded using target
        statistics from other folds, preventing target leakage.
        """
        if "denied" not in df.columns:
            raise ValueError(
                "Training DataFrame must contain a 'denied' column "
                "(binary target) for TargetEncoder fitting."
            )

        y = df["denied"].astype(int)

        # 75th percentile threshold for high_charge_claim
        self.high_charge_threshold_ = float(
            df["total_charge_amount"].quantile(0.75)
        )

        out = self._build_base_features(df)

        # TargetEncoder — fit_transform uses internal CV to avoid leakage
        cat_df = self._prepare_categorical_input(df)
        self.target_encoder_ = TargetEncoder(
            target_type="binary",
            smooth="auto",
            cv=5,
            random_state=42,
        )
        encoded = self.target_encoder_.fit_transform(cat_df, y)
        encoded_df = pd.DataFrame(
            encoded,
            columns=[f"{col}_encoded" for col in _CATEGORICAL_COLUMNS],
            index=df.index,
        )
        for col in encoded_df.columns:
            out[col] = encoded_df[col]

        self._is_fitted = True
        return self._finalize(out, df)

    # ---- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist fitted state to disk via joblib."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted FeatureEngineer.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": FEATURE_ENGINEERING_VERSION,
            "target_encoder": self.target_encoder_,
            "high_charge_threshold": self.high_charge_threshold_,
            "feature_columns": FEATURE_COLUMNS,
        }
        joblib.dump(state, path)
        logger.info("Saved FeatureEngineer state to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> FeatureEngineer:
        """Load a fitted FeatureEngineer from disk."""
        state = joblib.load(path)

        # Version check
        saved_version = state.get("version")
        if saved_version != FEATURE_ENGINEERING_VERSION:
            logger.warning(
                "FeatureEngineer version mismatch: saved=%s, current=%s",
                saved_version,
                FEATURE_ENGINEERING_VERSION,
            )

        # Feature column check
        saved_columns = state.get("feature_columns", [])
        if saved_columns != FEATURE_COLUMNS:
            logger.warning(
                "Feature columns mismatch: saved %d columns vs current %d columns",
                len(saved_columns),
                len(FEATURE_COLUMNS),
            )

        instance = cls()
        instance.target_encoder_ = state["target_encoder"]
        instance.high_charge_threshold_ = state["high_charge_threshold"]
        instance._is_fitted = True
        return instance

    # ---- metadata --------------------------------------------------------

    @staticmethod
    def get_feature_metadata() -> list[FeatureMeta]:
        """Return metadata for all output features."""
        return list(_FEATURE_METADATA)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_engineer: FeatureEngineer | None = None


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a raw claim DataFrame using a pre-fitted FeatureEngineer.

    Loads from the configured encoders path on first call, then reuses the
    instance for subsequent calls.
    """
    global _default_engineer
    if _default_engineer is None:
        _default_engineer = FeatureEngineer.load(settings.ML_ENCODERS_PATH)
    return _default_engineer.transform(df)
