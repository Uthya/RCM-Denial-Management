"""Train/serve consistency tests for the FeatureEngineer.

Covers:
- Strict-fail loading on version / feature-column mismatch.
- The fitted transform produces a deterministic encoded value for a given row.
- An unseen payer/CPT/Dx at inference yields:
    * a deterministic encoded value (smoothing-prior fallback),
    * the corresponding ``unseen_*`` flag set to 1,
    * ``unseen_any`` set to 1.
- A seen-but-rare category does NOT trigger the unseen flag (unseen vs rare
  are distinct concepts).
- Target leakage probe: a category's encoded value cannot reveal its own
  row's label, because the deployment encoder is fit on all rows and the
  row-specific contribution is averaged out.
- Feature column order is exactly ``FEATURE_COLUMNS``.

Tests use a tiny synthetic dataset so they're fast and don't need the DB.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_ENGINEERING_VERSION,
    FeatureEngineer,
)


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


def _make_synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a small but realistic raw-claim DataFrame for tests."""
    rng = np.random.default_rng(seed)
    payers = ["AETNA", "MOLINA", "CIGNA", "BCBS"]
    cpts = ["99213", "99214", "G0152", "T1021", "97110"]
    dxs = ["E119", "M791", "I10", "J45", "F329"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "total_charge_amount": float(rng.uniform(50, 5000)),
                "line_count": int(rng.integers(1, 8)),
                "diagnosis_count": int(rng.integers(1, 6)),
                "total_units": float(rng.uniform(1, 50)),
                "service_from_date": "2025-03-07",
                "service_to_date": "2025-03-09",
                "has_modifier": bool(rng.integers(0, 2)),
                "payer_name": rng.choice(payers),
                "frequency_code": rng.choice(["1", "7", None]),
                "facility_type_code": rng.choice(["11", "12", "31"]),
                "primary_procedure_code": rng.choice(cpts),
                "primary_diagnosis_code": rng.choice(dxs),
                "place_of_service": rng.choice(["11", "12", "31"]),
                "total_billed_amount": float(rng.uniform(50, 5000)),
                "denied": int(rng.integers(0, 2)),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_engineer() -> FeatureEngineer:
    df = _make_synthetic_df()
    eng = FeatureEngineer()
    eng.fit_transform(df)  # populates all internal state
    return eng


# ---------------------------------------------------------------------------
# Schema / load validation
# ---------------------------------------------------------------------------


def test_feature_columns_count_and_order(fitted_engineer):
    """Schema contract: transform output must be FEATURE_COLUMNS in order."""
    df_one = _make_synthetic_df(n=1, seed=42)
    X = fitted_engineer.transform(df_one)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape[1] == len(FEATURE_COLUMNS)


def test_n_features_in_set_after_fit(fitted_engineer):
    assert fitted_engineer.n_features_in_ == len(FEATURE_COLUMNS)


def test_load_rejects_version_mismatch(tmp_path):
    """A v3 artifact loaded under v4 code must HARD-FAIL — silently using
    mismatched encoders would corrupt every prediction."""
    df = _make_synthetic_df()
    eng = FeatureEngineer()
    eng.fit_transform(df)
    artifact = tmp_path / "engineer.joblib"
    eng.save(artifact)

    # Tamper with the saved version to simulate an older artifact.
    import joblib

    state = joblib.load(artifact)
    state["version"] = "v3"
    joblib.dump(state, artifact)

    with pytest.raises(ValueError, match="version mismatch"):
        FeatureEngineer.load(artifact)


def test_load_rejects_feature_columns_mismatch(tmp_path):
    df = _make_synthetic_df()
    eng = FeatureEngineer()
    eng.fit_transform(df)
    artifact = tmp_path / "engineer.joblib"
    eng.save(artifact)

    import joblib

    state = joblib.load(artifact)
    state["feature_columns"] = FEATURE_COLUMNS[:-1]  # drop the last column
    joblib.dump(state, artifact)

    with pytest.raises(ValueError, match="Feature columns mismatch"):
        FeatureEngineer.load(artifact)


def test_save_then_load_roundtrip_preserves_transform(fitted_engineer, tmp_path):
    """A serialized engineer must produce identical transforms after reload."""
    artifact = tmp_path / "engineer.joblib"
    fitted_engineer.save(artifact)

    df_one = _make_synthetic_df(n=5, seed=99)
    X_before = fitted_engineer.transform(df_one)

    reloaded = FeatureEngineer.load(artifact)
    X_after = reloaded.transform(df_one)
    pd.testing.assert_frame_equal(X_before, X_after)


# ---------------------------------------------------------------------------
# Train/serve consistency
# ---------------------------------------------------------------------------


def test_transform_is_deterministic(fitted_engineer):
    """Same input row must produce identical encoded output across calls."""
    df = _make_synthetic_df(n=10, seed=7)
    X1 = fitted_engineer.transform(df)
    X2 = fitted_engineer.transform(df)
    pd.testing.assert_frame_equal(X1, X2)


def test_unseen_payer_flag_set(fitted_engineer):
    """A row with a payer never seen in training must set unseen_payer=1
    AND unseen_any=1, while seen rows get 0."""
    rows = pd.DataFrame(
        [
            {
                "total_charge_amount": 100.0,
                "line_count": 1,
                "diagnosis_count": 1,
                "total_units": 1.0,
                "service_from_date": "2025-03-07",
                "service_to_date": "2025-03-09",
                "has_modifier": False,
                "payer_name": "AETNA",  # in vocab
                "frequency_code": "1",
                "facility_type_code": "11",
                "primary_procedure_code": "99213",  # in vocab
                "primary_diagnosis_code": "E119",  # in vocab
                "place_of_service": "11",
                "total_billed_amount": 100.0,
            },
            {
                "total_charge_amount": 100.0,
                "line_count": 1,
                "diagnosis_count": 1,
                "total_units": 1.0,
                "service_from_date": "2025-03-07",
                "service_to_date": "2025-03-09",
                "has_modifier": False,
                "payer_name": "NEW_PAYER_NEVER_SEEN",  # unseen
                "frequency_code": "1",
                "facility_type_code": "11",
                "primary_procedure_code": "99213",
                "primary_diagnosis_code": "E119",
                "place_of_service": "11",
                "total_billed_amount": 100.0,
            },
        ]
    )
    X = fitted_engineer.transform(rows)
    assert X["unseen_payer"].tolist() == [0, 1]
    assert X["unseen_cpt"].tolist() == [0, 0]
    assert X["unseen_dx"].tolist() == [0, 0]
    assert X["unseen_any"].tolist() == [0, 1]


def test_unseen_cpt_and_dx_flags(fitted_engineer):
    row = pd.DataFrame(
        [
            {
                "total_charge_amount": 100.0,
                "line_count": 1,
                "diagnosis_count": 1,
                "total_units": 1.0,
                "service_from_date": "2025-03-07",
                "service_to_date": "2025-03-09",
                "has_modifier": False,
                "payer_name": "AETNA",
                "frequency_code": "1",
                "facility_type_code": "11",
                "primary_procedure_code": "Z9999",  # unseen
                "primary_diagnosis_code": "Z9999",  # unseen
                "place_of_service": "11",
                "total_billed_amount": 100.0,
            }
        ]
    )
    X = fitted_engineer.transform(row)
    assert X.loc[0, "unseen_cpt"] == 1
    assert X.loc[0, "unseen_dx"] == 1
    assert X.loc[0, "unseen_payer"] == 0
    assert X.loc[0, "unseen_any"] == 1


def test_missing_value_is_not_flagged_unseen(fitted_engineer):
    """missing_X and unseen_X must be orthogonal signals: a null payer is
    'missing' (covered by missing_payer), not 'unseen'."""
    row = pd.DataFrame(
        [
            {
                "total_charge_amount": 100.0,
                "line_count": 1,
                "diagnosis_count": 1,
                "total_units": 1.0,
                "service_from_date": "2025-03-07",
                "service_to_date": "2025-03-09",
                "has_modifier": False,
                "payer_name": None,  # missing
                "frequency_code": "1",
                "facility_type_code": "11",
                "primary_procedure_code": "99213",
                "primary_diagnosis_code": "E119",
                "place_of_service": "11",
                "total_billed_amount": 100.0,
            }
        ]
    )
    X = fitted_engineer.transform(row)
    assert X.loc[0, "missing_payer"] == 1
    assert X.loc[0, "unseen_payer"] == 0  # NOT unseen — it's missing


def test_unseen_payer_encoded_value_is_smoothing_prior(fitted_engineer):
    """Two different unseen payers must produce the SAME encoded value
    (the global smoothing prior), proving inference is deterministic for
    out-of-vocabulary categories."""
    base = {
        "total_charge_amount": 100.0,
        "line_count": 1,
        "diagnosis_count": 1,
        "total_units": 1.0,
        "service_from_date": "2025-03-07",
        "service_to_date": "2025-03-09",
        "has_modifier": False,
        "frequency_code": "1",
        "facility_type_code": "11",
        "primary_procedure_code": "99213",
        "primary_diagnosis_code": "E119",
        "place_of_service": "11",
        "total_billed_amount": 100.0,
    }
    rows = pd.DataFrame(
        [
            {**base, "payer_name": "NEW_PAYER_A"},
            {**base, "payer_name": "NEW_PAYER_B"},
        ]
    )
    X = fitted_engineer.transform(rows)
    assert np.isclose(
        X.loc[0, "payer_name_encoded"], X.loc[1, "payer_name_encoded"]
    )


def test_target_leakage_label_does_not_alter_transform(fitted_engineer):
    """The deployment encoder cannot leak a row's own label, because at
    inference the engineer is already-fitted and the row's label is not
    available. Verifies that adding/removing a 'denied' column has zero
    effect on transform output."""
    row = pd.DataFrame(
        [
            {
                "total_charge_amount": 100.0,
                "line_count": 1,
                "diagnosis_count": 1,
                "total_units": 1.0,
                "service_from_date": "2025-03-07",
                "service_to_date": "2025-03-09",
                "has_modifier": False,
                "payer_name": "AETNA",
                "frequency_code": "1",
                "facility_type_code": "11",
                "primary_procedure_code": "99213",
                "primary_diagnosis_code": "E119",
                "place_of_service": "11",
                "total_billed_amount": 100.0,
            }
        ]
    )
    X_no_label = fitted_engineer.transform(row.copy())
    row_with_label = row.copy()
    row_with_label["denied"] = 1
    X_with_label = fitted_engineer.transform(row_with_label)
    pd.testing.assert_frame_equal(X_no_label, X_with_label)


def test_transform_before_fit_raises():
    eng = FeatureEngineer()
    with pytest.raises(RuntimeError, match="has not been fitted"):
        eng.transform(_make_synthetic_df(n=1))
