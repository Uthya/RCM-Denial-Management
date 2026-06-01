# Model Version Log

A running changelog of the denial-prediction model. **One entry per `MODEL_VERSION`.**
Each entry records what changed from the previous version, the benefit gained, and
the trade-offs (pros/cons), plus the metrics that justified the change.

Conventions:
- **Test metrics** = held-out split during training (stable, ~15k claims, ~29% denial rate).
- **Live metrics** = reconciled predictions from `prediction_log` (real 835 outcomes; smaller, ~16% denial rate). Live is the source of truth but noisier.
- `MODEL_VERSION` is set in `app/ml/trainer.py` and `app/ml/predictor.py`; predictions are tagged with it so versions can be compared in `/api/monitoring/live-performance?model_version=...`.

---

## v3.5 — Authorization / referral / provider-NPI / submission-lag features (FE v4 → v5)

**Date:** 2026-06-01 · **Status:** deployed (current)

**Trigger:** Feature audit identified five denial categories the model could not learn from existing data:
- CARC 15 / 95 / 197 / 198 — prior authorization absent or exceeded
- CARC 165 — referral missing
- CARC 38 / 170 / 185 / 242 — provider type / credentialing
- CARC 29 — timely-filing deadline missed
- CARC 110 — billing date predates service date

All of the required raw inputs were *already in the 837 stream* — they just weren't being captured by the parser or exposed to the feature engineer.

**Changes from v3.4:**

1. **Schema (Alembic `a1b2c3d4e5f6`).** Four nullable columns added to `claims`:
   `authorization_number`, `referral_number`, `billing_provider_npi`,
   `rendering_provider_npi`. Two indices on the NPI columns for per-fold
   target-encoding scans and OOV roll-up queries. Existing rows untouched —
   new columns are all `NULL` (which itself carries signal via the new
   presence flags).

2. **Parser (`services/parsers/handlers.py`).**
   - `handle_ref` extended to capture **REF*G1** / **REF*G3** (prior auth /
     predetermination) → `authorization_number`; **REF*9F** (referral) →
     `referral_number`.
   - `handle_nm1` extended to capture **NM1*85** (billing provider, Loop
     2010AA) and **NM1*82** (rendering provider, Loop 2310B), accepting the
     NPI from element 9 only when element 8's qualifier is `XX`. Other ID
     types (e.g. EIN with qualifier `24`) are explicitly rejected so the
     target encoder isn't poisoned with mixed identifier kinds.
   - Billing NPI is buffered on `ParseContext.current_billing_provider_npi`
     and consumed at CLM time (spec loop order places NM1*85 before CLM).

3. **Feature engineering bumped v4 → v5** (`app/ml/feature_engineering.py`).
   Seven new output features (35 → 42):
   - `has_prior_authorization` (boolean: REF*G1/G3 present + non-blank)
   - `has_referral`            (boolean: REF*9F present + non-blank)
   - `billing_provider_npi_encoded`     (TargetEncoder, 5-fold CV, leakage-safe)
   - `rendering_provider_npi_encoded`   (TargetEncoder, 5-fold CV, leakage-safe)
   - `unseen_billing_provider`          (1 if NPI not in training vocab)
   - `unseen_rendering_provider`        (1 if NPI not in training vocab)
   - `service_to_submission_days`       (int days, clipped >=0, 0 if null)

   The two NPI columns join `_CATEGORICAL_COLUMNS` so they ride the same
   per-fold-refit TargetEncoder regime as payer / CPT / Dx — no new
   leakage surface. Both NPI columns also join `_VOCABULARY_COLUMNS`, so
   the v4 unseen-flag machinery extends to providers, and `unseen_any`
   is now the OR across all five tracked dimensions (payer + CPT + Dx +
   billing-NPI + rendering-NPI).

4. **Submission-date sourcing.** `app.ml.dataset.build_dataset` reads
   `claim.created_at` (the TimestampMixin column populated when the 837 is
   ingested) as the submission timestamp. This is strictly before any
   835/denial outcome by construction → **no future-information leakage**.
   At inference, the predictor router passes the claim's `created_at` for
   stored claims, or `date.today()` for ad-hoc `/predict` calls.

5. **Defensive engineer changes.** `_prepare_categorical_input`,
   `_apply_unseen_flags`, and `_compute_category_vocabularies` now treat
   a source column missing from the input frame as "all-missing" rather
   than raising. Keeps the v5 engineer compatible with v4-shape callers
   that haven't been updated. Also fixes a pre-existing latent bug in
   `_stringify` where `None.astype(str)` yielded a float `NaN` (not the
   string `"nan"`) under pandas 2.x, which previously slipped past the
   sentinel replacement.

6. **Monitoring surfaces.**
   - `FEATURE_DISPLAY_NAMES` and `ML_FEATURE_HINTS` gained entries for
     all 7 new features (e.g. "Prior Authorization Present" → CARC 198
     hint; "New Billing Provider (no training history)" → credentialing
     advice).
   - `PredictionResponse.unseen_indicators` extended with optional
     `billing_provider` and `rendering_provider` fields (additive — older
     clients keep working).
   - `_SNAPSHOT_FIELDS` in `prediction_logger.py` extended so the v5
     inputs land in `prediction_log.feature_snapshot` and the
     `/unseen-rate` endpoint can roll up provider OOV.
   - `_UNSEEN_DIMENSIONS` in `routers/monitoring.py` extended with
     `billing_provider` / `rendering_provider` keys.
   - `distributions.py` bumped v1 → v2: provider NPIs added to the
     categorical PSI baseline, `service_to_submission_days` added to the
     continuous baseline, four new missingness sources tracked.

7. **MODEL_VERSION v3.4 → v3.5**, **FEATURE_VERSION v4 → v5**. Strict-fail
   loading still rejects mismatched artifacts, so a v3.4 model paired with
   v3.5 code refuses to serve.

**Expected impact (qualitative — empirical re-train pending on production data):**

| CARC category | Previously unlearnable | v5 surface |
|---|---|---|
| 15 / 95 / 197 / 198 (auth) | yes | `has_prior_authorization` |
| 165 (referral) | yes | `has_referral` |
| 38 / 170 / 185 / 242 (provider) | yes | `billing_provider_npi_encoded`, `rendering_provider_npi_encoded`, unseen-* flags |
| 29 / 110 / 146 (timely-filing / DOS) | yes | `service_to_submission_days` |

**Pros:** Five previously-blind denial categories now surface as direct
model features; provider-NPI OOV gets the same early-warning treatment as
new payers; submission-lag is computed from a strictly-pre-adjudication
timestamp (`claim.created_at`), so the timely-filing signal cannot leak;
all changes are backward-compatible with v4 input shape (defensive engineer
handles missing source columns by treating them as universally absent —
the existing 837 corpus continues to flow through the engineer unchanged).

**Cons:** Feature count grew (+7 columns, 35 → 42) and `n_features_in_`
permanently differs from v3.x / v4 — *no compatibility path back without a
code rollback*. The v5 features are most informative only on freshly-ingested
837s that include the new segments; the historic corpus parsed under
pre-v3.5 code carries all-NULL NPI / auth / referral columns and will
contribute only `service_to_submission_days` and the (zero) presence
flags until those claims are re-parsed.

**Tests (`tests/test_feature_engineering.py`):** 12 existing tests still
pass; 8 new tests added covering feature count (42), auth/referral flag
correctness, provider OOV flag orthogonality with missing values,
submission-lag arithmetic + clipping, backward-compatibility with v4
input shape, and a leakage probe confirming the deployment encoder produces
identical submission-lag features regardless of label presence. Smoke
test on a synthetic 837 carrying REF*G1, REF*9F, NM1*85, NM1*82 confirms
end-to-end capture into the Claim row.

---

## v3.4 — Training-pipeline optimizations (no scoring change; 155 s → 100 s full / 62 s warm)

**Date:** 2026-05-28 · **Status:** infra-only; v3.4 model unchanged

Infrastructure-only follow-up — no model semantics changed, no version bump. The deployed v3.4 model, calibration, thresholds, and risk buckets are byte-for-byte equivalent to what they were before. This entry documents the **training pipeline becoming faster**, not the model becoming different.

**Changes:**

1. **Pre-engineered CV fold cache during Optuna tuning.** The CV fold splits are deterministic (`StratifiedKFold(random_state=42)`) but the previous loop refit `FeatureEngineer` from scratch inside every trial's every fold — that work is identical across trials. Now the engineered fold matrices are computed once before the Optuna loop and reused across all trials. Leakage-safe by construction: each fold's engineer is fit on that fold's training rows only, same isolation as the prior `cross_val_score(Pipeline(FE, XGB))`.
2. **Adaptive Optuna trial budget.** `_adaptive_n_trials(N)` returns 5 / 10 / 20 / 30 trials for `N < 500 / < 5k / < 50k / ≥ 50k`. Honored when `n_trials` isn't explicitly passed.
3. **Early-stop callback on convergence.** Optuna study aborts when the best PR-AUC has not improved by ≥ `1e-3` over the last 5 trials. On this data TPE converges in trials 1–6; the rest was waste.
4. **Three retrain modes:** `full`, `warm`, `quick`, exposed via `--mode` CLI flag and `?mode=` query param on `/api/predictions/train`.
   - **Full**: Optuna search + 5-fold isotonic calibration. Default.
   - **Warm**: reuse last good hyperparameters from `training_metrics.json` (refuses if `MODEL_VERSION` or `FEATURE_ENGINEERING_VERSION` mismatch). Skip Optuna entirely.
   - **Quick**: warm + 3-fold calibration. For development.
5. **Per-stage timing instrumentation.** `_StageTimer` captures tuning / calibration / deploy_fit wall-clock; surfaced in `result["stage_timings_seconds"]` and the metrics artifact.
6. **Quality regression guard.** After every training, `_check_quality_regression` compares the new test metrics against the prior `training_metrics.json` and warns (does not block) on PR-AUC drop > 0.01, recall drop > 2 pp, or Brier rise > 0.005. Surfaced as `result["quality_check"]`.

**Benchmark on the production 177k dataset (same seed, same model):**

| Mode | Total time | Tuning | Calibration | Deploy fit | PR-AUC | ROC-AUC | Recall | Precision | Brier | Regression check |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (pre-fix) | ~155 s | ~120 s | ~20 s | ~10 s | 0.9220 | 0.9789 | 0.832 | 0.860 | 0.0264 | n/a |
| **Full (optimized)** | **99.9 s** | 34.1 s | 15.7 s | 3.7 s | 0.9220 | 0.9789 | 0.837 | 0.858 | 0.0264 | ok |
| **Warm** | **61.9 s** | 0.0 s | 15.3 s | 3.5 s | 0.9220 | 0.9789 | 0.837 | 0.858 | 0.0264 | ok |
| **Quick** | **58.4 s** | 0.0 s | 9.2 s | 4.0 s | 0.9219 | 0.9788 | 0.850 | 0.839 | 0.0264 | ok |

**Speedup:** 155 s → 100 s full retrain (**35% reduction**), 155 s → 62 s warm (**60% reduction**), 155 s → 58 s quick (**62% reduction**). All three modes pass the quality regression check.

**Pros:** Leakage-safe by construction (cache scoped per-fold, FE never sees val rows); zero model-quality regression at any mode; the warm/quick modes refuse to run with stale artifacts (`MODEL_VERSION` / `FEATURE_VERSION` mismatch falls back to full); per-stage timings make future optimization observable.
**Cons:** Cache adds ~150–250 MB peak memory during tuning (acceptable on any modern host); the early-stop patience (5) is tuned for this dataset's convergence — pathologically rugged loss landscapes could underspend (override via explicit `--n-trials`).

---

## v3.4 — Tighter LOW bucket + 5-fold calibration (LOW-but-denied: 1.63% → 0.87%)

**Date:** 2026-05-28 · **Status:** deployed (current)

**Trigger:** Operator-reported "many medium or low risk claims are being denied — medium is acceptable but low is not." Diagnostic on the 35,473-claim test split confirmed: the LOW bucket was emitting 1.63% real-world denial rate (492 of 30,154 LOW claims denied). The model's *calibration* was already honest (Brier 0.027; per-bin gap ≤4 pp across 10 bins) — the problem was the **bucket definition**, not the probabilities.

**Changes from v3.3:**
1. **LOW bucket detached from threshold and tightened.** Old rule: `LOW < 0.5 * τ` (~0.17); new rule: `LOW < 0.05` (fixed). MEDIUM widens to absorb the difference; HIGH unchanged (still `>= τ`). New constant `LOW_PROB_CUTOFF = 0.05` in `predictor.py`. `_classify_risk` updated.
2. **Calibration CV folds 3 → 5.** Finer-grained out-of-fold probability estimation, especially in the low-score region where ~85% of the test set lives. OOF Brier improved 0.037 → 0.026.

**Benefit gained (test split, 35,473 claims):**
| Bucket | n | denial rate v3.3 | denial rate v3.4 |
|---|---|---|---|
| LOW | 30,154 → 26,784 | **1.63%** | **0.87%** |
| MEDIUM | 1,022 → 4,392 | 24.6% | 11.6% |
| HIGH | 4,297 | 86.0% | 86.0% |

- **LOW-but-denied claims dropped 492 → 233** (53% reduction). "LOW = safe to ignore" now matches reality (~1 in 115 will deny).
- MEDIUM moved from "near-HIGH" (24.6%) to genuinely borderline (11.6%) — operationally useful as a "review-worthy" tier.
- HIGH semantics fully preserved (still `score >= τ`, same precision/recall trade-off).

**Pros:** Bucket labels now match operator intuition; no change to discrimination (ROC-AUC, PR-AUC, decision threshold all stable); 5-fold calibration gives marginally cleaner OOF probabilities; zero schema or API change.
**Cons:** MEDIUM bucket population grew ~4×, so dashboards showing MEDIUM counts will look very different in v3.4 (more claims flagged for review); this is intentional but worth communicating to billers.

**Hyperparameters / artifacts:** same Optuna-tuned XGBoost (`n_estimators=175, max_depth=10, scale_pos_weight≈16`), same precision-floor threshold strategy. Threshold landed at **0.3163** (within ~5% of v3.3's 0.3333 — minor float drift; not a semantics change).

**Metrics (test):**
| Recall | Precision | F1 | Accuracy | ROC-AUC | PR-AUC | Brier (cal) | Threshold |
|---|---|---|---|---|---|---|---|
| 0.832 | 0.860 | 0.846 | 0.962 | 0.979 | 0.922 | 0.027 | 0.3163 |

ROC-AUC ticked up 0.968 → 0.979 (the now-larger training set including QX/AB/NX added discrimination signal).

---

## v3.3 — ML correctness pass: CAS-parser fix, unseen-category flags, strict-fail artifacts

**Date:** 2026-05-28 · **Status:** superseded by v3.4

**Changes from v3.2:** This is a correctness-focused release. The model
architecture (XGBoost + isotonic calibration + precision-floor threshold)
is unchanged; data flowing into it improved, and the deployment surface got
hard contracts.

1. **CAS parser stride bug fixed** (`services/parsers/handlers.py`). The
   X12 005010 CAS triplet (reason, amount, quantity) was being mis-parsed
   for payers that omit empty quantity slots: the old stride-3 loop lost
   secondary triplets, mis-attributing denial dollars to wrong CARC codes.
   The new `_parse_cas_triplets` handles spec form, compact form, and
   mixed/malformed segments; quantity is now captured into the new
   `adjustments.quantity` column (alembic migration `b81a3f72c4d2`).
   **Impact:** CARC→dollar attribution is correct; recommendations driven
   by remittance adjustments are now trustworthy. 19 unit tests cover the
   forms (`tests/test_cas_parser.py`).

2. **Feature engineering bumped v3 → v4** with explicit unseen-category
   indicators (`feature_engineering.py`): `unseen_payer`, `unseen_cpt`,
   `unseen_dx`, `unseen_any`. Training-time vocabularies are persisted in
   the engineer artifact; at inference, an out-of-vocabulary payer/CPT/Dx
   now flags as "unseen" rather than collapsing into "low encoded value
   like a zero-denial seen value." Feature count grew 31 → 35.

3. **Strict-fail artifact loading.** `FeatureEngineer.load` raises
   `ValueError` (not a warning) on `feature_engineering_version` or
   `feature_columns` mismatch. `DenialPredictor.load` raises on
   `calibrator.model_version` ≠ `predictor.MODEL_VERSION`. A new
   `n_features_in_` check inside `predict()` guards against silent column
   drift. **No model serves under a mismatched artifact.**

4. **Feature schema artifact** persisted alongside the model
   (`app/ml/artifacts/feature_schema.json`). Captures
   `feature_engineering_version`, `feature_columns`, `n_features_in`,
   categorical input columns, training prevalence, calibration method, and
   the decision threshold. Used by integration tests and ops to verify
   deployment consistency without unpickling.

5. **Pre-existing single-row inference fragility fixed.**
   `_prepare_categorical_input` now uses the `__missing__` sentinel
   consistently (matching the joint encoder) instead of `np.nan`, so a
   single-row predict with a null categorical no longer crashes sklearn's
   `TargetEncoder` (which fails `np.isnan` on object-dtype categories when
   training had no nulls for that column). The `missing_*` flag features
   still carry the missingness signal explicitly to the model.

6. **12 train/serve consistency tests** (`tests/test_feature_engineering.py`)
   exercise version-mismatch refusal, transform determinism, unseen-vs-missing
   orthogonality, out-of-vocabulary determinism (two unseen payers produce
   the same smoothing-prior encoded value), and a target-leakage probe (the
   deployment encoder cannot leak a row's own label).

**Benefit gained:** Denial dollar attribution is correct. Inference under
nullable categoricals is deterministic. Train/serve schema drift now hard-
fails at load time instead of silently corrupting predictions. Unseen and
zero-denial categories — previously indistinguishable to the model — are
now separate signals. Retrained on 125,988 labelled claims.

**Pros:** Production correctness restored on a parser-level bug that was
silently corrupting denial analytics; hard contracts prevent the highest-
risk class of silent model failure (artifact/code drift); 31 automated tests
cover the previously-untested parsing and FE surfaces.
**Cons:** Feature count grew (+4 columns) and `n_features_in_` will
permanently differ from v3.x — *no compatibility path back to v3.x without
a code rollback*. The CAS parser still has a documented ambiguity at
6-element segments that could be either 2 spec triplets-with-quantity or 3
compact triplets; spec wins by default (see `_parse_cas_triplets` docstring).

**Metrics (test, held-out from 125,988):**
| Recall | Precision | F1 | Accuracy | ROC-AUC | PR-AUC | Brier (cal) | Threshold |
|---|---|---|---|---|---|---|---|
| 0.820 | 0.871 | 0.845 | 0.947 | 0.968 | 0.920 | 0.038 | 0.3333 |

Threshold landed at 0.3333 (precision-floor=0.85 strategy) — within
~1% of v3.2's 0.3246, confirming the precision-floor rule's stability
across the v3.3 feature/dataset changes.

**Addendum (2026-05-28, no version bump) — notification surfaces wired:**
The four v4 unseen-* features were only shifting the model's predictions;
they weren't surfaced to users or ops. Three surfaces now make that signal
visible:
1. **Per-claim**: `PredictionResponse.unseen_indicators` (`payer`/`cpt`/`dx`/`any`)
   exposed by `/api/predictions/predict` and `/predict-claim`;
   `ClaimPrediction.unseen_any` exposed by `/predict-file`. The
   `ClaimDetailPage` renders an amber banner when any dimension is unseen.
2. **Contribution text**: `FEATURE_DISPLAY_NAMES` and `ML_FEATURE_HINTS`
   gained entries for the four flags ("New Payer (no training history)" with
   a biller-meaningful hint), so `top_risk_factors` and the recommendation
   engine produce readable copy when the flags drive a score.
3. **Aggregate**: new `GET /api/monitoring/unseen-rate?days=N&top_n=M`
   endpoint computes the OOV rate from `prediction_log.feature_snapshot`
   against the deployed engineer's `category_vocabularies_`. A new
   `UnseenRateCard` on the Monitoring page surfaces overall rate +
   per-dimension counts + top-N unseen values. Verified end-to-end with a
   synthetic OOV claim (all three dims flagged true and aggregated in the
   1-day window).

---

## v3.2 — Precision-floor balanced threshold

**Date:** 2026-05-27 · **Status:** superseded by v3.3

**Changes from v3.1:**
1. Threshold strategy F2 → **precision-floor** (`THRESHOLD_STRATEGY="precision_floor"`,
   `PRECISION_FLOOR=0.85`): the highest-recall threshold whose precision still clears
   the floor. Robust to the train/live prevalence gap that broke F2. Landed
   threshold = **0.3246** (auto-derived, not hardcoded).
2. Same calibration + threshold-tied buckets as v3.1 (HIGH ≥ τ, MEDIUM ≥ 0.5τ).
3. **Retrained on the expanded dataset** — 125,888 labelled claims (was 76,404) after
   ingesting the 50k external claim-pairs (originals + replacements). Denial prevalence
   fell 29% → 17.6%, so the tuned `scale_pos_weight` rose to ~10.6 (calibration corrects).

**Why:** F2 (~0.19) was too aggressive in production — precision collapsed to 0.39 on
live data because the threshold was tuned on higher-prevalence training data. The
precision-floor rule targets a false-alarm budget directly, so it transfers across the
prevalence gap.

**Benefit gained:** Balanced operating point — recall 0.83 / precision 0.87 on test,
the best precision/recall balance of any usable version. Calibration improved further
(test Brier 0.079 → 0.038).

**Pros:** Fixes v3.1's false-alarm flood while keeping high recall; threshold derived
from a controllable precision budget rather than a blind F-beta; trained on 65% more data.
**Cons:** Threshold still derived from *training* prevalence (precision-floor makes this
far more robust, but a live-derived threshold is still the ideal follow-up); these 50k
claims are now training data, so a fresh unseen batch is needed for the next clean eval.

**Metrics (test, ~31k held-out from the 125,888):**
| Recall | Precision | F1 | Accuracy | ROC-AUC | PR-AUC | Brier (cal) | Threshold |
|---|---|---|---|---|---|---|---|
| 0.829 | 0.869 | 0.849 | 0.948 | 0.969 | 0.920 | 0.038 | 0.3246 |

**External-eval validation (50k unseen claim pairs, before ingestion):** at threshold
0.30 the shared model scored recall 0.70 / precision 0.88 — best balanced of all four
versions tested (v2 0.60/0.99, v3 0.66/0.999, v3.1 0.79/0.46, v3.2 0.70/0.88).

---

## v3.1 — Recall-favouring threshold (F2) + threshold-tied risk buckets

**Date:** 2026-05-27 · **Status:** superseded by v3.2

**Changes from v3:**
1. Decision-threshold strategy F1 → **F2** (`FBETA=2.0`) — weights recall 2× precision.
   Threshold moved 0.474 → **0.19**.
2. `risk_level` buckets **tied to the threshold** instead of fixed 0.3/0.7:
   HIGH ≥ τ, MEDIUM ≥ 0.5τ, LOW < 0.5τ. (`_classify_risk` in `predictor.py`.)

**Benefit gained:** Catches far more denials (test recall 0.81 → 0.90; live recall
0.62 → 0.86). "LOW-but-denied" rate dropped to its lowest (live 7.4% → 2.9%) because
LOW now means a genuinely low calibrated probability.

**Pros:** Highest recall; HIGH now means "predicted to deny" (coherent with the label);
buckets meaningful on the calibrated scale.
**Cons:** **Precision collapsed in production** (live 1.00 → 0.39 — ~6 of 10 flags are
false alarms) because the F2 threshold was tuned on higher-prevalence training data.
Too aggressive for live; superseded by the v3.2 proposal.

**Metrics:**
| | Recall | Precision | ROC-AUC | PR-AUC | Brier (cal) |
|---|---|---|---|---|---|
| Test | 0.897 | 0.785 | 0.953 | 0.937 | 0.0516 |
| Live (n=129, 21 denied) | 0.857 | 0.391 | — | — | — |

---

## v3 — Calibration + tuned threshold + PR-AUC objective + tuned scale_pos_weight

**Date:** 2026-05-27 · **Status:** superseded by v3.1

**Changes from v2:**
1. **Isotonic calibration** of probabilities (leakage-free, out-of-fold; mirrors
   `CalibratedClassifierCV(ensemble=False)`). `risk_score` is now a true probability.
2. **Tuned decision threshold** (Step 1) via F1 on out-of-fold calibrated scores → 0.474
   (replaced the hardcoded 0.5).
3. **Optuna objective ROC-AUC → PR-AUC** (average precision) — imbalance-aware (Step 2).
   ROC-AUC still reported.
4. **`scale_pos_weight` tuned** as a search parameter around the natural ratio (Step 3);
   landed ~5.57 vs natural 2.45.
5. New artifact `calibrator.joblib` (calibrator + threshold). Predictor applies
   calibration, emits `predicted_label`/`decision_threshold`. `MODEL_VERSION` → v3.

**Benefit gained:** Honest probabilities (test Brier 0.080 → 0.051, ~36% better).
Live recall improved 0.42 → 0.62 with perfect live precision (1.00).

**Pros:** Trustworthy calibrated scores; leakage-free CV/calibration; better live
recall and precision than v2 simultaneously.
**Cons:** F1 threshold is precision-heavy → still missed 38% of denials live; the
`risk_level` buckets were NOT yet realigned to the calibrated scale (fixed in v3.1).

**Metrics:**
| | Recall | Precision | ROC-AUC | PR-AUC | Brier (cal) |
|---|---|---|---|---|---|
| Test | 0.810 | 0.962 | 0.955 | 0.939 | 0.0513 |
| Live (n=121, 21 denied) | 0.619 | 1.000 | — | — | — |

---

## v2 — XGBoost baseline (+ mid-life leakage fix & Optuna tuning)

**Date:** ≤ 2026-05-27 · **Status:** superseded by v3

**Defining characteristics:** XGBoost on the v3 feature set (TargetEncoder + joint
denial-rate features). Uncalibrated raw scores used directly as `risk_score`; fixed
0.5 decision threshold; fixed `risk_level` buckets at 0.3 / 0.7; `scale_pos_weight`
fixed at the natural class ratio.

**Changes landed within v2's lifespan (2026-05-27, still tagged v2):**
- **Leakage fix:** feature engineering moved *inside* the CV pipeline and the raw frame
  split *before* fitting, so target encoders never see held-out rows. (Foundation that
  made v3's honest metrics possible.)
- **Optuna hyperparameter tuning** added (objective: ROC-AUC at this point).

**Pros:** Strong ranking ability (test ROC-AUC ~0.955); simple, no calibration moving parts.
**Cons:** Uncalibrated, inflated scores (meaningless as probabilities); arbitrary 0.5
threshold → poor live recall (0.42 — missed 58% of denials); pre-fix versions had
target leakage inflating offline metrics.

**Metrics:**
| | Recall | Precision | ROC-AUC |
|---|---|---|---|
| Test (leakage-free run) | ~0.85 | ~0.90 | ~0.955 |
| Live (n=737, 144 denied) | 0.424 | 0.897 | — |

---

### How to update this log
When `MODEL_VERSION` changes (or a material model change ships), add a new entry at the
top with: change-from-previous, benefit gained, pros/cons, and test + live metrics.
Keep the newest version first.
