"""Feature engineering for claim-level denial prediction.

Transforms the raw DataFrame from ``dataset.build_dataset()`` into a
31-feature numeric matrix suitable for XGBoost training and inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder

from app.core.config import settings

logger = logging.getLogger(__name__)

FEATURE_ENGINEERING_VERSION = "v4"
# v3 -> v4: explicit unseen-category indicators (payer/CPT/Dx/any) and a
# persisted training-time category vocabulary, so inference can distinguish
# "category never seen in training" from "category seen but never denied."

# Sentinel string used when stringifying nullable categorical values for joint
# keys. Keeps the joint cardinality meaningful (e.g. "ACME||missing") instead
# of producing NaN-containing keys that TargetEncoder would treat oddly.
_MISSING_SENTINEL = "__missing__"

# Volume thresholds for rarity flags. Tuned for the current dataset; the
# specific numbers matter less than the *concept* — they tell the model when
# a category is statistically thin and the TargetEncoder estimate is noisy.
_RARE_PAYER_THRESHOLD = 50
_RARE_CPT_THRESHOLD = 10
_RARE_DX_THRESHOLD = 10

# Joint keys whose denial rate the model should see directly. Pairs are
# encoded as a synthetic string column "value_a||value_b" then run through a
# dedicated TargetEncoder with the same 5-fold CV used for single columns.
_JOINT_KEYS: list[tuple[str, str]] = [
    ("payer_name", "primary_procedure_code"),
    ("payer_name", "primary_diagnosis_code"),
    ("payer_name", "place_of_service"),
]

# Columns whose training-time frequency the model should see directly. Rare
# values get small volume counts → the model can learn to trust the encoded
# denial rate less for them.
_VOLUME_COLUMNS: list[str] = [
    "payer_name",
    "primary_procedure_code",
    "primary_diagnosis_code",
]

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
    # Joint / cross-key denial-rate aggregations (v3) — give the model the
    # denial rate of (payer × CPT), (payer × dx), (payer × POS) pairs
    # directly, instead of forcing tree splits to discover the interaction.
    "payer_cpt_denial_rate",
    "payer_dx_denial_rate",
    "payer_pos_denial_rate",
    # Volume / reliability features (v3) — training-time counts so the model
    # can downweight encoded denial rates from sparsely-observed categories.
    "payer_volume",
    "cpt_volume",
    "dx_volume",
    "is_rare_payer",
    "is_rare_cpt",
    # Unseen-at-training indicators (v4). Distinguish "we've never seen this
    # value in training" from "we've seen it but it had zero denials" — both
    # used to produce identical low encoded values, blinding the model to a
    # very different operational situation.
    "unseen_payer",
    "unseen_cpt",
    "unseen_dx",
    "unseen_any",
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
    FeatureMeta("payer_cpt_denial_rate", "float64", "payer_name × primary_procedure_code", "joint TargetEncoder"),
    FeatureMeta("payer_dx_denial_rate", "float64", "payer_name × primary_diagnosis_code", "joint TargetEncoder"),
    FeatureMeta("payer_pos_denial_rate", "float64", "payer_name × place_of_service", "joint TargetEncoder"),
    FeatureMeta("payer_volume", "int64", "payer_name", "training count"),
    FeatureMeta("cpt_volume", "int64", "primary_procedure_code", "training count"),
    FeatureMeta("dx_volume", "int64", "primary_diagnosis_code", "training count"),
    FeatureMeta("is_rare_payer", "int64", "payer_name", f"payer_volume < {_RARE_PAYER_THRESHOLD}"),
    FeatureMeta("is_rare_cpt", "int64", "primary_procedure_code", f"cpt_volume < {_RARE_CPT_THRESHOLD}"),
    FeatureMeta("unseen_payer", "int64", "payer_name", "value not in training-time vocabulary"),
    FeatureMeta("unseen_cpt", "int64", "primary_procedure_code", "value not in training-time vocabulary"),
    FeatureMeta("unseen_dx", "int64", "primary_diagnosis_code", "value not in training-time vocabulary"),
    FeatureMeta("unseen_any", "int64", "payer/cpt/dx", "any of unseen_payer/cpt/dx is 1"),
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

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transforms a raw claim DataFrame into the numeric feature matrix.

    A proper scikit-learn transformer (``BaseEstimator`` + ``TransformerMixin``)
    so it can be dropped into a ``Pipeline`` and cloned/refit inside each
    cross-validation fold. Fitting per fold is what keeps the target encoders
    from ever seeing validation/test labels — the leakage-free guarantee the
    tuning pipeline relies on.
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
        "payer_volume",
        "cpt_volume",
        "dx_volume",
        "is_rare_payer",
        "is_rare_cpt",
        "unseen_payer",
        "unseen_cpt",
        "unseen_dx",
        "unseen_any",
    ]

    # Source columns whose vocabularies we persist for the unseen-* flags.
    _VOCABULARY_COLUMNS: ClassVar[dict[str, str]] = {
        "unseen_payer": "payer_name",
        "unseen_cpt": "primary_procedure_code",
        "unseen_dx": "primary_diagnosis_code",
    }

    def __init__(self) -> None:
        self.target_encoder_: TargetEncoder | None = None
        self.joint_target_encoder_: TargetEncoder | None = None
        self.high_charge_threshold_: float | None = None
        # Per-category training-set counts, keyed by source column name.
        # Used at inference to surface "how often did we see this value"
        # without having to re-scan the training set.
        self.volume_lookups_: dict[str, dict[str, int]] | None = None
        # Training-time vocabulary per source column (v4). Used at inference
        # to set unseen_* flags so the model can distinguish "never seen"
        # from "seen but never denied" — those used to look identical.
        self.category_vocabularies_: dict[str, frozenset[str]] | None = None
        self.n_features_in_: int | None = None
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
        """Extract categorical columns as a string DataFrame for TargetEncoder.

        Missing values are stringified to the ``__missing__`` sentinel rather
        than left as ``NaN``. Reason: sklearn's TargetEncoder ``transform``
        path crashes (``isnan`` on object-dtype categories_) when training
        data had no NaN for a column but inference does. Treating missing as
        a category aligns with the joint encoder and makes single-row
        inference with nullable categoricals deterministic; the dedicated
        ``missing_*`` flag features still carry the missingness signal
        explicitly to the model.
        """
        cat_df = df[_CATEGORICAL_COLUMNS].copy()
        for col in _CATEGORICAL_COLUMNS:
            cat_df[col] = self._stringify(cat_df[col])
        return cat_df

    @staticmethod
    def _stringify(series: pd.Series) -> pd.Series:
        """Stringify a column, replacing nulls with a fixed sentinel.

        Used when constructing joint keys — null values must collapse to a
        consistent token so e.g. ``"ACME||{missing}"`` is one category, not a
        scattered set of distinct keys.
        """
        return (
            series.astype(str)
            .replace("nan", _MISSING_SENTINEL)
            .replace("None", _MISSING_SENTINEL)
        )

    def _build_joint_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the synthetic joint-key DataFrame consumed by the joint encoder."""
        out = pd.DataFrame(index=df.index)
        for col_a, col_b in _JOINT_KEYS:
            name = f"{col_a}||{col_b}"
            out[name] = self._stringify(df[col_a]) + "||" + self._stringify(df[col_b])
        return out

    def _compute_volume_lookups(self, df: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Precompute training-time value counts for each tracked column."""
        lookups: dict[str, dict[str, int]] = {}
        for col in _VOLUME_COLUMNS:
            counts = self._stringify(df[col]).value_counts()
            lookups[col] = counts.to_dict()
        return lookups

    def _compute_category_vocabularies(
        self, df: pd.DataFrame
    ) -> dict[str, frozenset[str]]:
        """Snapshot the set of distinct non-missing categorical values per source
        column. Missing values are EXCLUDED so they're handled by the missing_*
        flags, not conflated with unseen-at-training.
        """
        vocab: dict[str, frozenset[str]] = {}
        for src_col in self._VOCABULARY_COLUMNS.values():
            stringified = self._stringify(df[src_col])
            values = {v for v in stringified.unique() if v != _MISSING_SENTINEL}
            vocab[src_col] = frozenset(values)
        return vocab

    def _apply_unseen_flags(self, out: pd.DataFrame, df: pd.DataFrame) -> None:
        """Append unseen_payer / unseen_cpt / unseen_dx / unseen_any flags.

        ``unseen_X`` is 1 when the row's value for source column X is non-missing
        AND wasn't in the training-time vocabulary. Missing values are NOT
        flagged unseen (they're covered by missing_X), so the two signals stay
        independent.
        """
        assert self.category_vocabularies_ is not None, "vocabularies must be fitted"
        unseen_any = pd.Series(0, index=df.index, dtype="int64")
        for feat_name, src_col in self._VOCABULARY_COLUMNS.items():
            stringified = self._stringify(df[src_col])
            seen = self.category_vocabularies_.get(src_col, frozenset())
            # value is unseen if it's not missing AND not in the seen vocab.
            mask = (stringified != _MISSING_SENTINEL) & (~stringified.isin(seen))
            flag = mask.astype("int64")
            out[feat_name] = flag
            unseen_any = (unseen_any | flag).astype("int64")
        out["unseen_any"] = unseen_any

    def _apply_joint_encoding(
        self, out: pd.DataFrame, df: pd.DataFrame, *, is_training: bool, y=None
    ) -> None:
        """Append the three joint denial-rate columns to ``out``.

        During training (``is_training=True``) we call ``fit_transform`` on the
        joint encoder so each row gets out-of-fold target rates (matching the
        leakage-safe behaviour of the per-column TargetEncoder). At inference
        we call ``transform`` only.
        """
        joint_df = self._build_joint_input(df)
        if is_training:
            self.joint_target_encoder_ = TargetEncoder(
                target_type="binary",
                smooth="auto",
                cv=5,
                random_state=42,
            )
            encoded = self.joint_target_encoder_.fit_transform(joint_df, y)
        else:
            assert self.joint_target_encoder_ is not None
            encoded = self.joint_target_encoder_.transform(joint_df)

        # Maintain a stable column order: payer_cpt, payer_dx, payer_pos.
        out["payer_cpt_denial_rate"] = encoded[:, 0]
        out["payer_dx_denial_rate"] = encoded[:, 1]
        out["payer_pos_denial_rate"] = encoded[:, 2]

    def _apply_volume_features(self, out: pd.DataFrame, df: pd.DataFrame) -> None:
        """Append per-category training-count columns + rarity flags to ``out``."""
        assert self.volume_lookups_ is not None
        payer_counts = self.volume_lookups_.get("payer_name", {})
        cpt_counts = self.volume_lookups_.get("primary_procedure_code", {})
        dx_counts = self.volume_lookups_.get("primary_diagnosis_code", {})

        payer_vol = self._stringify(df["payer_name"]).map(payer_counts).fillna(0)
        cpt_vol = self._stringify(df["primary_procedure_code"]).map(cpt_counts).fillna(0)
        dx_vol = self._stringify(df["primary_diagnosis_code"]).map(dx_counts).fillna(0)

        out["payer_volume"] = payer_vol.astype("int64")
        out["cpt_volume"] = cpt_vol.astype("int64")
        out["dx_volume"] = dx_vol.astype("int64")
        out["is_rare_payer"] = (payer_vol < _RARE_PAYER_THRESHOLD).astype("int64")
        out["is_rare_cpt"] = (cpt_vol < _RARE_CPT_THRESHOLD).astype("int64")

    # ---- fit / transform / fit_transform --------------------------------

    @staticmethod
    def _resolve_target(df: pd.DataFrame, y) -> np.ndarray:
        """Return the binary target as an int array.

        Accepts an explicit ``y`` (scikit-learn ``fit(X, y)`` convention, used
        by the CV pipeline) or falls back to a ``denied`` column on ``df`` for
        direct callers. Converting to a positionally-aligned array avoids any
        index-label surprises when sklearn hands us a fold subset.
        """
        if y is None:
            if "denied" not in df.columns:
                raise ValueError(
                    "Target not provided: pass y to fit()/fit_transform() or "
                    "include a 'denied' column."
                )
            y = df["denied"]
        return np.asarray(y).astype(int)

    def fit(self, X: pd.DataFrame, y=None) -> FeatureEngineer:
        """Fit encoders and thresholds from the training data.

        The target may be passed explicitly as ``y`` (sklearn convention) or
        read from a ``denied`` column on ``X``.
        """
        df = X
        y = self._resolve_target(df, y)

        # TargetEncoder — fit on all 6 categorical columns at once
        cat_df = self._prepare_categorical_input(df)
        self.target_encoder_ = TargetEncoder(
            target_type="binary",
            smooth="auto",
            cv=5,
            random_state=42,
        )
        self.target_encoder_.fit(cat_df, y)

        # Joint-key TargetEncoder for cross-column denial rates.
        joint_df = self._build_joint_input(df)
        self.joint_target_encoder_ = TargetEncoder(
            target_type="binary",
            smooth="auto",
            cv=5,
            random_state=42,
        )
        self.joint_target_encoder_.fit(joint_df, y)

        # Volume lookups for reliability features.
        self.volume_lookups_ = self._compute_volume_lookups(df)

        # Category vocabularies for unseen-* flags (v4).
        self.category_vocabularies_ = self._compute_category_vocabularies(df)

        # 75th percentile threshold for high_charge_claim
        self.high_charge_threshold_ = float(
            df["total_charge_amount"].quantile(0.75)
        )

        self.n_features_in_ = len(FEATURE_COLUMNS)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Produce the full feature matrix from a raw DataFrame."""
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer has not been fitted. Call fit() first.")

        out = self._build_base_features(df)

        # Categorical target encoding (deployment mapping — produces the
        # values inference will see for these rows).
        cat_df = self._prepare_categorical_input(df)
        encoded = self.target_encoder_.transform(cat_df)
        encoded_df = pd.DataFrame(
            encoded,
            columns=[f"{col}_encoded" for col in _CATEGORICAL_COLUMNS],
            index=df.index,
        )
        for col in encoded_df.columns:
            out[col] = encoded_df[col]

        # Joint denial-rate aggregations + volume / reliability features +
        # unseen-at-training indicators.
        self._apply_joint_encoding(out, df, is_training=False)
        self._apply_volume_features(out, df)
        self._apply_unseen_flags(out, df)

        return self._finalize(out, df)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step.

        Uses TargetEncoder.fit_transform() which applies internal cross-fitting
        (5-fold CV) so each training row is encoded using target statistics from
        *other* folds. Combined with refitting inside each outer CV fold (via
        ``Pipeline``), validation/test rows never influence their own encoding.

        The target may be passed explicitly as ``y`` (sklearn convention) or
        read from a ``denied`` column on ``X``.
        """
        df = X
        y = self._resolve_target(df, y)

        # 75th percentile threshold for high_charge_claim
        self.high_charge_threshold_ = float(
            df["total_charge_amount"].quantile(0.75)
        )

        # Volume lookups (target-independent, so compute up front).
        self.volume_lookups_ = self._compute_volume_lookups(df)

        # Category vocabularies for unseen-* flags (v4).
        self.category_vocabularies_ = self._compute_category_vocabularies(df)

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

        # Joint denial-rate aggregations (also CV-folded internally).
        self._apply_joint_encoding(out, df, is_training=True, y=y)

        # Volume / rarity features + unseen-at-training indicators.
        self._apply_volume_features(out, df)
        self._apply_unseen_flags(out, df)

        self.n_features_in_ = len(FEATURE_COLUMNS)
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
            "joint_target_encoder": self.joint_target_encoder_,
            "volume_lookups": self.volume_lookups_,
            "category_vocabularies": self.category_vocabularies_,
            "high_charge_threshold": self.high_charge_threshold_,
            "feature_columns": FEATURE_COLUMNS,
            "n_features_in": self.n_features_in_,
        }
        joblib.dump(state, path)
        logger.info("Saved FeatureEngineer state to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> FeatureEngineer:
        """Load a fitted FeatureEngineer from disk.

        Strict-fail on version or feature-column mismatch: a v3 artifact paired
        with v4 code (or vice versa) would produce silently wrong feature
        vectors at inference, so refuse to load rather than warn.
        """
        state = joblib.load(path)

        saved_version = state.get("version")
        if saved_version != FEATURE_ENGINEERING_VERSION:
            raise ValueError(
                f"FeatureEngineer version mismatch: saved={saved_version!r}, "
                f"current={FEATURE_ENGINEERING_VERSION!r}. Retrain the model — "
                "loading an artifact built under a different feature-engineering "
                "version risks silently-wrong predictions."
            )

        saved_columns = state.get("feature_columns", [])
        if saved_columns != FEATURE_COLUMNS:
            raise ValueError(
                f"Feature columns mismatch: saved {len(saved_columns)} cols, "
                f"current {len(FEATURE_COLUMNS)} cols. Retrain to regenerate "
                f"the artifact."
            )

        instance = cls()
        instance.target_encoder_ = state["target_encoder"]
        instance.joint_target_encoder_ = state.get("joint_target_encoder")
        instance.volume_lookups_ = state.get("volume_lookups")
        instance.category_vocabularies_ = state.get("category_vocabularies")
        instance.high_charge_threshold_ = state["high_charge_threshold"]
        instance.n_features_in_ = state.get("n_features_in", len(FEATURE_COLUMNS))
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
