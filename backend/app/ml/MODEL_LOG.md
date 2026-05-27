# Model Version Log

A running changelog of the denial-prediction model. **One entry per `MODEL_VERSION`.**
Each entry records what changed from the previous version, the benefit gained, and
the trade-offs (pros/cons), plus the metrics that justified the change.

Conventions:
- **Test metrics** = held-out split during training (stable, ~15k claims, ~29% denial rate).
- **Live metrics** = reconciled predictions from `prediction_log` (real 835 outcomes; smaller, ~16% denial rate). Live is the source of truth but noisier.
- `MODEL_VERSION` is set in `app/ml/trainer.py` and `app/ml/predictor.py`; predictions are tagged with it so versions can be compared in `/api/monitoring/live-performance?model_version=...`.

---

## v3.2 — Precision-floor balanced threshold

**Date:** 2026-05-27 · **Status:** deployed (current)

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
