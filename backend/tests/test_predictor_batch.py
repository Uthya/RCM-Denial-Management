"""Batch-equivalence regression tests for ``DenialPredictor.predict_batch``.

The ``/predict-file`` router calls ``predict_batch`` once for the whole
file instead of looping ``predict`` per-claim — a ~100x speedup. Per-row
output MUST equal what a non-batched, per-claim scoring loop would
produce; otherwise we silently corrupt every prediction in production.

This file pins that contract. The strategy is to compute per-claim
predictions through an **independent reference scorer defined in this
test file** (mirroring the pre-batching ``predict()`` exactly), then
assert the production ``predict_batch`` output matches field-for-field.

Why an independent reference matters: ``DenialPredictor.predict`` now
delegates to ``predict_batch([claim])[0]``. Asserting ``predict() ==
predict_batch([c])[0]`` would be a tautology — a regression that breaks
both paths together would pass. The reference scorer below is a fresh
implementation that pins what "correct per-claim output" means.

The fixture builds a fully-functioning predictor (feature engineer +
XGBoost + isotonic calibrator) entirely in-memory on the v5-shaped
synthetic dataset. No DB and no on-disk artifacts — CI-safe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from app.ml.feature_engineering import (
    FEATURE_COLUMNS,
    FeatureEngineer,
)
from app.ml.predictor import (
    FEATURE_DISPLAY_NAMES,
    LOW_PROB_CUTOFF,
    DenialPredictor,
)
from app.services.prediction_logger import _row_from_prediction
from app.services.recommendations import build_recommendations_for_claim


# ---------------------------------------------------------------------------
# Synthetic data — v5-shaped (auth / referral / NPIs / submission_date)
# ---------------------------------------------------------------------------


def _make_training_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Training-shape DataFrame with all v5 input columns populated."""
    rng = np.random.default_rng(seed)
    payers = ["AETNA", "MOLINA", "CIGNA", "BCBS"]
    cpts = ["99213", "99214", "G0152", "T1021", "97110"]
    dxs = ["E119", "M791", "I10", "J45", "F329"]
    billing_npis = ["1111111111", "2222222222", "3333333333"]
    rendering_npis = ["4444444444", "5555555555", "6666666666"]
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
                "authorization_number": (
                    f"AUTH-{i}" if rng.integers(0, 2) else None
                ),
                "referral_number": (
                    f"REF-{i}" if rng.integers(0, 3) == 0 else None
                ),
                "billing_provider_npi": rng.choice(billing_npis),
                "rendering_provider_npi": rng.choice(rendering_npis),
                "submission_date": "2025-03-20",
                "denied": int(rng.integers(0, 2)),
            }
        )
    return pd.DataFrame(rows)


def _make_claim_dicts(n: int = 30, seed: int = 7) -> list[dict]:
    """Inference-shape claim dicts — same schema as ``_claim_to_predict_dict``
    in the router. Includes:
      * "normal" claims with seen categorical values
      * claims with an unseen payer (exercises ``unseen_payer`` flag)
      * claims with an unseen billing NPI (v5)
      * claims missing optional fields
      * a claim with no authorization / referral / submission_date
    """
    rng = np.random.default_rng(seed)
    payers_seen = ["AETNA", "MOLINA", "CIGNA", "BCBS"]
    cpts_seen = ["99213", "99214", "G0152", "T1021", "97110"]
    dxs_seen = ["E119", "M791", "I10", "J45", "F329"]
    bnp_seen = ["1111111111", "2222222222", "3333333333"]
    rnp_seen = ["4444444444", "5555555555", "6666666666"]

    claims: list[dict] = []
    for i in range(n):
        base = {
            "total_charge_amount": float(rng.uniform(50, 5000)),
            "line_count": int(rng.integers(1, 5)),
            "diagnosis_count": int(rng.integers(1, 4)),
            "total_units": float(rng.uniform(1, 30)),
            "service_from_date": "2025-08-01",
            "service_to_date": "2025-08-02",
            "has_modifier": bool(rng.integers(0, 2)),
            "payer_name": rng.choice(payers_seen),
            "frequency_code": "1",
            "facility_type_code": "11",
            "primary_procedure_code": rng.choice(cpts_seen),
            "primary_diagnosis_code": rng.choice(dxs_seen),
            "place_of_service": "11",
            "total_billed_amount": float(rng.uniform(50, 5000)),
            "authorization_number": f"AUTH-{i}" if i % 2 == 0 else None,
            "referral_number": f"REF-{i}" if i % 3 == 0 else None,
            "billing_provider_npi": rng.choice(bnp_seen),
            "rendering_provider_npi": rng.choice(rnp_seen),
            "submission_date": "2025-08-15",
        }
        # Mix in edge-case shapes that the production loader produces
        # (unseen categoricals, missing fields).
        if i == 5:
            base["payer_name"] = "NEWLY_ONBOARDED_PAYER"  # unseen
        if i == 7:
            base["billing_provider_npi"] = "9999999999"  # unseen NPI
        if i == 11:
            base["rendering_provider_npi"] = "8888888888"  # unseen NPI
        if i == 13:
            base["authorization_number"] = None
            base["referral_number"] = None
            base["submission_date"] = None
        if i == 17:
            base["payer_name"] = None
            base["primary_diagnosis_code"] = None
        claims.append(base)
    return claims


# ---------------------------------------------------------------------------
# Fully-fitted predictor fixture (model + engineer + calibrator, in-memory)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_predictor() -> DenialPredictor:
    """Build a DenialPredictor entirely in-memory.

    - FeatureEngineer fit on the v5 synthetic dataset (so vocabularies
      include the known categorical values; OOV claims later trigger the
      unseen-* flag paths).
    - Tiny XGBoost trained on those features (10 trees, depth 3 — enough
      for tree-SHAP to produce non-trivial per-row contributions).
    - Isotonic calibrator fit on the model's in-sample raw scores. The
      calibrator math isn't being validated here — we only need it
      present so the calibration branch of predict_batch is exercised.

    The DenialPredictor instance is hand-assembled rather than loaded
    from disk: no model.json / encoders.joblib / calibrator.joblib
    artifacts are touched, so the test runs in any CI environment.
    """
    df = _make_training_df()
    y = df["denied"]

    engineer = FeatureEngineer()
    X = engineer.fit_transform(df, y)

    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
        n_jobs=1,
    )
    model.fit(X, y)

    raw_scores = model.predict_proba(X)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_scores, np.asarray(y).astype(int))

    predictor = DenialPredictor()
    predictor.model = model
    predictor.engineer = engineer
    predictor.calibrator = calibrator
    predictor.threshold = 0.30
    predictor._loaded = True
    return predictor


@pytest.fixture(scope="module")
def synthetic_claims() -> list[dict]:
    return _make_claim_dicts()


# ---------------------------------------------------------------------------
# Independent reference scorer — pinned in the test file
# ---------------------------------------------------------------------------


def _reference_classify_risk(score: float, threshold: float) -> str:
    """Inlined risk-bucket logic. Pins the contract independently of
    ``predictor._classify_risk`` so a future change to that helper can't
    silently invalidate this test."""
    if score >= threshold:
        return "HIGH"
    if score >= LOW_PROB_CUTOFF:
        return "MEDIUM"
    return "LOW"


def _reference_predict_one(predictor: DenialPredictor, claim: dict) -> dict:
    """Reference per-claim scorer.

    Mirrors the pre-batching ``DenialPredictor.predict()`` exactly:
    1-row DataFrame → FE.transform → model.predict_proba → calibrator →
    one ``booster.predict(pred_contribs=True)`` call → top-5 sort → per-
    column unseen flags read. Returns only the deterministic fields; the
    caller is responsible for excluding ``prediction_id`` / timestamp.

    The whole point is that this function is structurally INDEPENDENT of
    ``predictor.predict_batch``. They share artifacts (engineer, model,
    calibrator, threshold) but not code.
    """
    df = pd.DataFrame([claim])
    features = predictor.engineer.transform(df)

    raw_score = float(predictor.model.predict_proba(features)[0, 1])
    if predictor.calibrator is not None:
        cal = float(predictor.calibrator.transform(np.array([raw_score]))[0])
        risk_score = float(min(max(cal, 0.0), 1.0))
    else:
        risk_score = raw_score

    # SHAP contributions for this single row only.
    booster = predictor.model.get_booster()
    dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
    contrib_values = booster.predict(dmatrix, pred_contribs=True)[0, :-1]

    abs_sum = float(np.abs(contrib_values).sum())
    if abs_sum == 0:
        abs_sum = 1.0
    rows = []
    for i, col in enumerate(FEATURE_COLUMNS):
        val = float(contrib_values[i])
        pct = (val / abs_sum) * 100
        rows.append(
            {
                "feature": FEATURE_DISPLAY_NAMES.get(col, col),
                "impact": f"{'+' if val >= 0 else ''}{pct:.0f}%",
                "direction": "risk" if val >= 0 else "protective",
                "_abs": abs(val),
            }
        )
    rows.sort(key=lambda x: x["_abs"], reverse=True)
    top5 = [{k: v for k, v in r.items() if k != "_abs"} for r in rows[:5]]

    unseen_cols = (
        "unseen_payer", "unseen_cpt", "unseen_dx",
        "unseen_billing_provider", "unseen_rendering_provider", "unseen_any",
    )
    unseen_indicators = None
    if all(c in features.columns for c in unseen_cols):
        r0 = features.iloc[0]
        unseen_indicators = {
            "payer": bool(int(r0["unseen_payer"])),
            "cpt": bool(int(r0["unseen_cpt"])),
            "dx": bool(int(r0["unseen_dx"])),
            "billing_provider": bool(int(r0["unseen_billing_provider"])),
            "rendering_provider": bool(int(r0["unseen_rendering_provider"])),
            "any": bool(int(r0["unseen_any"])),
        }

    return {
        "risk_score": round(risk_score, 4),
        "raw_risk_score": round(raw_score, 4),
        "predicted_label": int(risk_score >= predictor.threshold),
        "decision_threshold": round(predictor.threshold, 4),
        "risk_level": _reference_classify_risk(risk_score, predictor.threshold),
        "top_risk_factors": top5,
        "unseen_indicators": unseen_indicators,
    }


# Fields that must match byte-for-byte between reference and batched paths.
# prediction_id (uuid4) and prediction_timestamp (now()) are intentionally
# excluded — they're stamped per-call and are not part of the scoring
# contract.
_DETERMINISTIC_FIELDS: tuple[str, ...] = (
    "risk_score",
    "raw_risk_score",
    "predicted_label",
    "decision_threshold",
    "risk_level",
    "top_risk_factors",
    "unseen_indicators",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_predict_batch_matches_reference_per_claim(
    fitted_predictor: DenialPredictor, synthetic_claims: list[dict]
) -> None:
    """The production ``predict_batch`` output for each claim must equal
    the independent reference scorer's output for that same claim.

    This is the load-bearing test: any future refactor that breaks
    per-row semantics in the batched path (e.g. accidentally couples
    rows in SHAP, swaps a per-row column read for a per-batch broadcast,
    silently reorders results) fails here, with a precise per-claim
    per-field error message."""
    batched = fitted_predictor.predict_batch(synthetic_claims)
    assert len(batched) == len(synthetic_claims), "batch length mismatch"

    mismatches: list[str] = []
    for i, claim in enumerate(synthetic_claims):
        ref = _reference_predict_one(fitted_predictor, claim)
        batch_row = batched[i]
        for field in _DETERMINISTIC_FIELDS:
            if ref[field] != batch_row.get(field):
                mismatches.append(
                    f"claim[{i}].{field}: reference={ref[field]!r} "
                    f"batched={batch_row.get(field)!r}"
                )

    assert not mismatches, (
        f"{len(mismatches)} per-claim field mismatch(es) between reference "
        f"and predict_batch:\n  " + "\n  ".join(mismatches[:10])
    )


def test_predict_single_delegates_to_batch(
    fitted_predictor: DenialPredictor, synthetic_claims: list[dict]
) -> None:
    """``predict(c)`` and ``predict_batch([c])[0]`` are the same code path
    today (the former is a thin wrapper). Asserting equality documents
    that contract — if a future refactor splits them, this test forces
    explicit consideration of why."""
    for claim in synthetic_claims[:5]:
        single = fitted_predictor.predict(claim)
        batched = fitted_predictor.predict_batch([claim])[0]
        for field in _DETERMINISTIC_FIELDS:
            assert single[field] == batched[field], (
                f"predict() vs predict_batch([c])[0] differ on {field}: "
                f"{single[field]!r} vs {batched[field]!r}"
            )


def test_predict_batch_preserves_input_order(
    fitted_predictor: DenialPredictor,
) -> None:
    """Results[i] must correspond to claims[i]. The router's zip() over
    (claims, pred_inputs, preds) relies on this; if predict_batch ever
    reordered (e.g. by grouping similar claims internally for vector
    efficiency), every claim_id↔risk_score pairing in /predict-file
    would silently corrupt."""
    # Build claims with distinct, learnable signal so the order is
    # detectable from the output (high charge → higher raw score on this
    # toy model).
    base = _make_claim_dicts(n=1)[0]
    claims = []
    for charge in (50.0, 5000.0, 100.0, 4500.0, 250.0):
        c = dict(base)
        c["total_charge_amount"] = charge
        c["total_billed_amount"] = charge
        claims.append(c)

    batched = fitted_predictor.predict_batch(claims)
    # Compare each position against the same row scored individually —
    # the position MUST match.
    for i, claim in enumerate(claims):
        single = fitted_predictor.predict(claim)
        assert batched[i]["raw_risk_score"] == single["raw_risk_score"], (
            f"order broken at index {i}: batched raw={batched[i]['raw_risk_score']}, "
            f"single raw={single['raw_risk_score']}"
        )


def test_predict_batch_empty_input_returns_empty_list(
    fitted_predictor: DenialPredictor,
) -> None:
    """Edge case: empty claim list → empty result list, no exception. The
    router would otherwise hit a 0-row pandas/XGBoost call which is a
    well-known crash site in older versions."""
    assert fitted_predictor.predict_batch([]) == []


def test_predict_batch_recommendation_output_matches_reference(
    fitted_predictor: DenialPredictor, synthetic_claims: list[dict]
) -> None:
    """``build_recommendations_for_claim`` is a pure function of
    ``top_risk_factors``. If the batched path produced even subtly
    different factor strings (e.g. ``"+0%"`` vs ``"+0.0%"``), the
    recommendation output would diverge silently. Catch that here."""
    batched = fitted_predictor.predict_batch(synthetic_claims)
    for i, claim in enumerate(synthetic_claims):
        ref = _reference_predict_one(fitted_predictor, claim)
        ref_recs = build_recommendations_for_claim(
            ml_factors=ref["top_risk_factors"]
        )
        batch_recs = build_recommendations_for_claim(
            ml_factors=batched[i]["top_risk_factors"]
        )
        assert ref_recs == batch_recs, (
            f"claim[{i}]: recommendation output differs between reference "
            f"and batched paths"
        )


def test_predict_batch_prediction_log_payload_matches_reference(
    fitted_predictor: DenialPredictor, synthetic_claims: list[dict]
) -> None:
    """``_row_from_prediction`` builds the DB row the prediction_log
    table receives. Every persisted column except prediction_id and
    prediction_time must match between the reference and batched paths —
    otherwise the live-performance / drift dashboards would compute on
    different inputs depending on which code path was used."""
    batched = fitted_predictor.predict_batch(synthetic_claims)
    # _row_from_prediction needs the full result dict (incl.
    # prediction_id / timestamp) but we don't compare those.
    deterministic_cols = (
        "claim_id", "claim_number", "predicted_risk", "predicted_label",
        "risk_level", "model_version", "feature_engineering_version",
        "feature_snapshot",
    )
    for i, claim in enumerate(synthetic_claims):
        ref_result = _reference_predict_one(fitted_predictor, claim)
        # The reference scorer omits version + prediction_id / timestamp;
        # synthesize the missing keys so _row_from_prediction can be
        # called identically against both inputs.
        ref_payload = {
            **ref_result,
            "prediction_id": "00000000-0000-0000-0000-000000000000",
            "model_version": batched[i]["model_version"],
            "feature_version": batched[i]["feature_version"],
            "prediction_timestamp": batched[i]["prediction_timestamp"],
        }
        row_ref = _row_from_prediction(
            claim_id=i, claim_number=f"TEST-{i}",
            claim_input=claim, prediction_result=ref_payload,
        )
        row_batch = _row_from_prediction(
            claim_id=i, claim_number=f"TEST-{i}",
            claim_input=claim, prediction_result=batched[i],
        )
        for col in deterministic_cols:
            assert getattr(row_ref, col) == getattr(row_batch, col), (
                f"claim[{i}] prediction_log.{col}: reference="
                f"{getattr(row_ref, col)!r} batched="
                f"{getattr(row_batch, col)!r}"
            )


def test_predict_batch_unseen_indicators_set_correctly(
    fitted_predictor: DenialPredictor,
) -> None:
    """Spot-check that the batched unseen-flag wiring still works: at
    least one OOV dimension should fire on the curated OOV claims, and
    none on a claim with fully-seen values. Defensive backstop in case
    a future change accidentally pulls indicators from the wrong row."""
    claims = _make_claim_dicts()  # claim[5]=unseen payer, [7]=unseen billing NPI, [11]=unseen rendering NPI
    batched = fitted_predictor.predict_batch(claims)

    # Fully seen claim (index 0 has all canonical values).
    seen = batched[0]["unseen_indicators"]
    assert seen is not None
    assert seen["payer"] is False
    assert seen["billing_provider"] is False
    assert seen["rendering_provider"] is False

    # Unseen payer.
    assert batched[5]["unseen_indicators"]["payer"] is True
    assert batched[5]["unseen_indicators"]["any"] is True

    # Unseen billing NPI.
    assert batched[7]["unseen_indicators"]["billing_provider"] is True
    assert batched[7]["unseen_indicators"]["any"] is True

    # Unseen rendering NPI.
    assert batched[11]["unseen_indicators"]["rendering_provider"] is True
    assert batched[11]["unseen_indicators"]["any"] is True


def test_predict_batch_handles_calibrator_disabled(
    fitted_predictor: DenialPredictor,
) -> None:
    """If the calibrator is absent (older artifact path), risk_score
    should equal raw_risk_score. Exercises the ``self.calibrator is
    None`` branch in predict_batch."""
    # Build a fresh predictor sharing the model + engineer, but with no
    # calibrator. Don't mutate the module-scoped fixture.
    bare = DenialPredictor()
    bare.model = fitted_predictor.model
    bare.engineer = fitted_predictor.engineer
    bare.calibrator = None
    bare.threshold = 0.5
    bare._loaded = True

    claims = _make_claim_dicts(n=3)
    out = bare.predict_batch(claims)
    for row in out:
        assert row["risk_score"] == row["raw_risk_score"]
