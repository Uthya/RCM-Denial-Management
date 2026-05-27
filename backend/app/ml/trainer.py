"""XGBoost training pipeline for claim denial prediction.

Trains an XGBClassifier on the feature-engineered dataset, evaluates
performance, and persists the model + metrics as artifacts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sqlalchemy.ext.asyncio import AsyncSession
from xgboost import XGBClassifier

from app.core.config import settings
from app.ml.dataset import build_dataset, get_dataset_stats
from app.ml.distributions import (
    build_distribution_snapshot,
    save_distribution_snapshot,
)
from app.ml.feature_engineering import FEATURE_COLUMNS, FeatureEngineer
from app.models.training_metric import TrainingMetric

# v2 -> v3: calibrated probabilities + tuned decision threshold (not 0.5).
# v3 -> v3.1: recall-favouring threshold (F2) + risk buckets tied to threshold.
# v3.1 -> v3.2: F2 (~0.19) was too aggressive in production (precision collapsed
# under the train/live prevalence gap). v3.2 uses a precision-floor threshold
# (~0.30) — the balanced operating point validated on the internal test set,
# live reconciled data, AND a 50k external eval. Bumping the version keeps each
# scoring regime separate in the monitoring layer.
MODEL_VERSION = "v3.2"

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TRAINING_SAMPLES = 50

# Hyperparameter tuning defaults
TUNE_BY_DEFAULT = True
DEFAULT_N_TRIALS = 30
TUNING_TIMEOUT_SECONDS = 180
MAX_CV_FOLDS = 5
# PR-AUC (average precision) is the search OBJECTIVE — imbalance-aware, unlike
# ROC-AUC. It rewards the search for ranking the minority (denied) class well.
# NOTE: ROC-AUC is still computed and reported in metrics; only the metric the
# search optimises changed.
TUNING_METRIC = "average_precision"

# scale_pos_weight is searched over this multiplier range around the natural
# class ratio (neg/pos) rather than fixed at it — the optimal weight is rarely
# exactly the ratio.
SPW_SEARCH_LO = 0.5
SPW_SEARCH_HI = 3.0

# Out-of-fold calibration + threshold selection settings.
CALIBRATION_METHOD = "isotonic"
CALIBRATION_CV_FOLDS = 3
# Decision-threshold selection on out-of-fold calibrated probabilities.
#   "precision_floor" — highest-recall threshold with precision >= PRECISION_FLOOR.
#                       Catches as many denials as possible within a false-alarm
#                       budget; robust to the train/live prevalence gap. (v3.2)
#   "fbeta"           — maximise F-beta; beta>1 favours recall. (v3 used F1, v3.1 F2)
THRESHOLD_STRATEGY = "precision_floor"
PRECISION_FLOOR = 0.85   # on training OOF (~29% prevalence) lands threshold ~0.30
FBETA = 2.0              # only used when THRESHOLD_STRATEGY == "fbeta"
DEFAULT_THRESHOLD = 0.5

# Quiet Optuna's per-trial chatter; we log our own summary instead.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _default_hyperparameters(scale_pos_weight: float) -> dict:
    """Baseline XGBoost hyperparameters used when tuning is disabled."""
    return {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "scale_pos_weight": round(scale_pos_weight, 4),
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
    }


def _tune_hyperparameters(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    n_trials: int,
    timeout: int | None,
) -> dict:
    """Search XGBoost hyperparameters with Optuna via stratified k-fold CV.

    Each trial cross-validates a full ``FeatureEngineer -> XGBClassifier``
    pipeline on the *raw* training frame. Because the feature engineer is
    refit inside every fold, the target encoders only ever see that fold's
    training labels — the cross-validated score is leakage-free, and the
    held-out test set is never touched here.

    Parameters
    ----------
    df_train : pd.DataFrame
        Raw (un-engineered) training split. Feature engineering happens
        inside the CV pipeline, per fold.
    y_train : pd.Series
        Binary labels aligned with ``df_train`` (used for stratification and
        passed through the pipeline to the encoders/model).
    scale_pos_weight
        Class-imbalance weight (held fixed across trials; derived from the
        full label distribution).
    n_trials
        Maximum number of Optuna trials.
    timeout
        Wall-clock budget in seconds (``None`` for no limit).

    Returns
    -------
    dict
        ``{"hyperparameters": <full param dict>, "tuning": <study summary>}``.
    """
    # Stratified CV needs at least one positive per fold; clamp folds to the
    # minority-class count so tiny/imbalanced datasets don't crash the study.
    minority_count = int(min(y_train.sum(), len(y_train) - y_train.sum()))
    n_folds = max(2, min(MAX_CV_FOLDS, minority_count))

    if minority_count < 2:
        # Not enough minority samples to cross-validate meaningfully; fall
        # back to the baseline configuration rather than tuning on noise.
        logger.warning(
            "Skipping hyperparameter tuning: only %d minority-class sample(s) "
            "in the training split",
            minority_count,
        )
        params = _default_hyperparameters(scale_pos_weight)
        return {
            "hyperparameters": params,
            "tuning": {
                "enabled": True,
                "skipped_reason": "insufficient_minority_samples",
                "n_trials_completed": 0,
            },
        }

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            # Step 3: tune the class-imbalance weight around the natural ratio
            # instead of fixing it. Calibration (downstream) restores honest
            # probabilities after whatever weight the search prefers.
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                scale_pos_weight * SPW_SEARCH_LO,
                scale_pos_weight * SPW_SEARCH_HI,
            ),
        }
        # Feature engineering lives INSIDE the pipeline so cross_val_score
        # refits the encoders on each fold's training rows only — no leakage
        # from the held-out fold into its own encoded features.
        pipeline = Pipeline(
            [
                ("features", FeatureEngineer()),
                (
                    "model",
                    XGBClassifier(
                        **params,
                        random_state=RANDOM_STATE,
                        eval_metric="logloss",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        scores = cross_val_score(
            pipeline, df_train, y_train, cv=cv, scoring=TUNING_METRIC, n_jobs=1
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    start = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    elapsed = round(time.time() - start, 3)

    best = study.best_params
    logger.info(
        "Optuna best %s=%.4f after %d trial(s) in %.1fs; params=%s",
        TUNING_METRIC,
        study.best_value,
        len(study.trials),
        elapsed,
        best,
    )

    # ``best`` already carries the tuned scale_pos_weight — do not override it.
    params = {
        **best,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
    }
    tuning = {
        "enabled": True,
        "metric": TUNING_METRIC,
        "natural_scale_pos_weight": round(scale_pos_weight, 4),
        "tuned_scale_pos_weight": round(float(best["scale_pos_weight"]), 4),
        "best_cv_score": round(float(study.best_value), 4),
        "n_trials_requested": n_trials,
        "n_trials_completed": len(study.trials),
        "cv_folds": n_folds,
        "timeout_seconds": timeout,
        "tuning_time_seconds": elapsed,
        "best_params": best,
    }
    return {"hyperparameters": params, "tuning": tuning}


def _select_threshold(
    y_true,
    probs,
    strategy: str = THRESHOLD_STRATEGY,
    beta: float = FBETA,
    precision_floor: float = PRECISION_FLOOR,
) -> float:
    """Pick the decision threshold on calibrated probabilities (y_true, probs).

    The 0.5 default is arbitrary on an imbalanced, cost-asymmetric problem, so
    we scan the precision-recall curve. Two strategies:

    - ``"precision_floor"``: the lowest threshold (→ highest recall) whose
      precision is still >= ``precision_floor``. Catches as many denials as
      possible within a false-alarm budget, and is robust to the train/live
      prevalence gap that made a pure recall-chasing threshold misbehave.
    - ``"fbeta"``: maximise F-beta; ``beta>1`` weights recall over precision.

    Operates on calibrated probabilities so the chosen value sits on a
    meaningful probability scale and transfers across calibrators.
    """
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return DEFAULT_THRESHOLD

    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    # precision/recall have one more element than thresholds; align by dropping
    # the last point (which corresponds to recall=0 / no threshold).
    p, r, t = precision[:-1], recall[:-1], thresholds
    if len(t) == 0:
        return DEFAULT_THRESHOLD

    if strategy == "precision_floor":
        ok = np.where(p >= precision_floor)[0]
        if len(ok):
            # Among thresholds meeting the floor, take the one with highest
            # recall (the lowest qualifying threshold).
            best = float(t[ok[int(np.argmax(r[ok]))]])
        else:
            # Floor unattainable on this data — fall back to the max-precision
            # point rather than flagging everything.
            best = float(t[int(np.argmax(p))])
    else:  # "fbeta"
        b2 = beta * beta
        denom = (b2 * p) + r
        fbeta = np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)
        best = float(t[int(np.argmax(fbeta))])

    # Guard against degenerate 0.0/1.0 thresholds.
    return min(max(best, 1e-4), 1 - 1e-4)


def _build_pipeline(hyperparameters: dict) -> Pipeline:
    """A FeatureEngineer -> XGBClassifier pipeline for the given params."""
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            ("model", XGBClassifier(n_jobs=-1, **hyperparameters)),
        ]
    )


def _fit_calibrator_and_threshold(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameters: dict,
) -> dict:
    """Fit an isotonic calibrator + decision threshold, leakage-free.

    Mirrors ``CalibratedClassifierCV(ensemble=False)``: out-of-fold predictions
    over the training split give honest (un-leaked) scores, the isotonic
    regressor is fit on those, and the threshold is chosen on the calibrated
    OOF scores. The base model itself is refit on all of ``df_train`` afterward
    by the caller, so the calibrator matches a model trained on the same data.

    Isotonic (not Platt) because: (1) ~tens of thousands of samples make
    overfitting a non-issue, and (2) the tuned scale_pos_weight distorts
    probabilities in a non-sigmoidal way that only a non-parametric monotonic
    fit corrects well.
    """
    n_folds = max(2, min(CALIBRATION_CV_FOLDS, int(min(y_train.sum(), len(y_train) - y_train.sum()))))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    pipeline = _build_pipeline(hyperparameters)
    oof_raw = cross_val_predict(
        pipeline, df_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(oof_raw, np.asarray(y_train).astype(int))
    oof_cal = calibrator.transform(oof_raw)

    threshold = _select_threshold(y_train, oof_cal)

    # Calibration quality on the OOF scores (lower Brier = better calibrated).
    yt = np.asarray(y_train).astype(int)
    brier_before = round(float(brier_score_loss(yt, oof_raw)), 5)
    brier_after = round(float(brier_score_loss(yt, oof_cal)), 5)
    logger.info(
        "Calibration (OOF): Brier %.5f -> %.5f; threshold=%.4f (strategy=%s, folds=%d)",
        brier_before,
        brier_after,
        threshold,
        THRESHOLD_STRATEGY,
        n_folds,
    )

    return {
        "calibrator": calibrator,
        "threshold": threshold,
        "info": {
            "method": CALIBRATION_METHOD,
            "cv_folds": n_folds,
            "threshold": round(threshold, 4),
            "threshold_strategy": THRESHOLD_STRATEGY,
            "precision_floor": PRECISION_FLOOR if THRESHOLD_STRATEGY == "precision_floor" else None,
            "oof_brier_before": brier_before,
            "oof_brier_after": brier_after,
        },
    }


def _run_training(
    df: pd.DataFrame,
    tune: bool = TUNE_BY_DEFAULT,
    n_trials: int = DEFAULT_N_TRIALS,
    tuning_timeout: int | None = TUNING_TIMEOUT_SECONDS,
) -> dict:
    """Train an XGBClassifier and return results with metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw labelled DataFrame from ``build_dataset()``.
    tune : bool
        When ``True``, search hyperparameters with Optuna (cross-validated on
        the training split). When ``False``, use the fixed baseline params.
    n_trials : int
        Maximum Optuna trials when ``tune`` is enabled.
    tuning_timeout : int | None
        Wall-clock budget for tuning in seconds (``None`` for no limit).

    Returns
    -------
    dict
        Training results including metrics, feature importance, and artifact paths.

    Raises
    ------
    ValueError
        If ``df`` has fewer than ``MIN_TRAINING_SAMPLES`` rows.
    """
    if len(df) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_TRAINING_SAMPLES} labelled samples for training, "
            f"got {len(df)}"
        )

    start_time = time.time()

    # Log class distribution
    stats = get_dataset_stats(df)
    logger.info("Dataset stats: %s", stats)

    # Label + class-balance weight (computed on the full dataset).
    y = df["denied"]
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    logger.info(
        "Class balance — positive: %d, negative: %d, scale_pos_weight: %.3f",
        n_positive,
        n_negative,
        scale_pos_weight,
    )

    # Split the RAW frame BEFORE any feature engineering. This is the core
    # leakage fix: target encoders are fit only on training rows, never on the
    # held-out test rows they are later scored against. (stratify only when
    # both classes are present.)
    stratify = y if y.nunique() > 1 else None
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    # Hyperparameters — tuned with Optuna (leakage-free CV on df_train) or the
    # fixed baseline. ``scale_pos_weight`` is always carried through so the
    # final model accounts for class imbalance regardless of tuning.
    if tune:
        logger.info(
            "Tuning hyperparameters with Optuna (n_trials=%d, timeout=%ss)",
            n_trials,
            tuning_timeout,
        )
        tune_result = _tune_hyperparameters(
            df_train, y_train, scale_pos_weight, n_trials, tuning_timeout
        )
        hyperparameters = tune_result["hyperparameters"]
        tuning_info = tune_result["tuning"]
    else:
        hyperparameters = _default_hyperparameters(scale_pos_weight)
        tuning_info = {"enabled": False}

    # --- Deployed model: fit on the TRAIN split ----------------------------
    # The engineer + model are fit on df_train only; df_test stays untouched
    # for honest evaluation below. We deploy this train-split model (reserving
    # 20% for unbiased metrics) rather than refitting on everything, so the
    # calibrator below matches the exact model that produced its OOF scores.
    engineer = FeatureEngineer()
    X_train = engineer.fit_transform(df_train, y_train)
    model = XGBClassifier(n_jobs=-1, **hyperparameters)
    model.fit(X_train, y_train)

    # --- Calibration + decision threshold (Step 1 + calibration) -----------
    # Leakage-free: isotonic calibrator and F1-optimal threshold are derived
    # from out-of-fold predictions over df_train (df_test is never involved).
    cal_result = _fit_calibrator_and_threshold(df_train, y_train, hyperparameters)
    calibrator = cal_result["calibrator"]
    threshold = cal_result["threshold"]
    calibration_info = cal_result["info"]

    # --- Honest evaluation on the untouched test split ----------------------
    # Raw model score -> calibrated probability -> label at the tuned threshold.
    # This is the exact serving path, so reported metrics describe production.
    X_test = engineer.transform(df_test)
    raw_test = model.predict_proba(X_test)[:, 1]
    cal_test = calibrator.transform(raw_test)
    y_pred = (cal_test >= threshold).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(
            float(precision_score(y_test, y_pred, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    if y_test.nunique() > 1:
        try:
            # Ranking metrics are invariant to the monotonic calibrator, so
            # ROC-AUC here matches the raw model's ranking. PR-AUC added too.
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, cal_test)), 4)
            metrics["pr_auc"] = round(
                float(average_precision_score(y_test, cal_test)), 4
            )
        except ValueError:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
            logger.warning("Could not compute ROC-AUC / PR-AUC")
        # Calibration quality on the held-out test split (lower Brier = better).
        metrics["brier_uncalibrated"] = round(
            float(brier_score_loss(y_test, raw_test)), 5
        )
        metrics["brier_calibrated"] = round(
            float(brier_score_loss(y_test, cal_test)), 5
        )
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        logger.warning("Skipping ROC-AUC/PR-AUC: only one class in test set")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    confusion = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    logger.info("Test metrics: %s", metrics)
    logger.info("Confusion matrix (threshold=%.4f): %s", threshold, confusion)

    # Feature importance (from the deployed model)
    importances = dict(
        zip(FEATURE_COLUMNS, model.feature_importances_.tolist(), strict=False)
    )
    importances = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)
    )
    top_10 = list(importances.items())[:10]
    logger.info("Top 10 features: %s", top_10)

    # Save artifacts
    model_path = Path(settings.ML_MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    logger.info("Saved model to %s", model_path)

    encoders_path = Path(settings.ML_ENCODERS_PATH)
    encoders_path.parent.mkdir(parents=True, exist_ok=True)
    engineer.save(encoders_path)
    logger.info("Saved encoders to %s", encoders_path)

    # Save the calibrator + decision threshold together (one artifact). The
    # predictor loads this to map raw scores -> calibrated probability and to
    # label at the tuned threshold instead of a hardcoded 0.5.
    calibrator_path = Path(settings.ML_CALIBRATOR_PATH)
    calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "method": CALIBRATION_METHOD,
            "calibrator": calibrator,
            "threshold": float(threshold),
            "model_version": MODEL_VERSION,
        },
        calibrator_path,
    )
    logger.info("Saved calibrator + threshold to %s", calibrator_path)

    trained_at = datetime.now(timezone.utc).isoformat()
    training_time = round(time.time() - start_time, 3)

    # Write training metrics JSON
    metrics_path = Path(settings.ML_METRICS_PATH)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_artifact = {
        "metrics": metrics,
        "feature_importance": importances,
        "hyperparameters": hyperparameters,
        "tuning": tuning_info,
        "calibration": calibration_info,
        "dataset_size": len(df),
        "trained_at": trained_at,
    }
    metrics_path.write_text(json.dumps(metrics_artifact, indent=2))
    logger.info("Saved training metrics to %s", metrics_path)

    # Persist a lightweight training-data distribution snapshot for the
    # drift monitor. Best-effort: a snapshot failure must not abort
    # training (the model + encoders are already on disk by this point).
    try:
        # Use CALIBRATED training scores so the drift monitor's score baseline
        # matches what production now emits (calibrated probabilities).
        training_scores = calibrator.transform(model.predict_proba(X_train)[:, 1])
        snapshot = build_distribution_snapshot(
            raw_df=df_train,
            feature_df=X_train,
            training_scores=training_scores,
            trained_at=trained_at,
            feature_version="v3",
            model_version=MODEL_VERSION,
        )
        save_distribution_snapshot(snapshot, settings.ML_DISTRIBUTIONS_PATH)
    except Exception:
        logger.exception("Failed to write training distribution snapshot")

    return {
        "status": "success",
        "trained_at": trained_at,
        "training_time_seconds": training_time,
        "dataset_stats": stats,
        "class_balance": {
            "n_positive": n_positive,
            "n_negative": n_negative,
            "scale_pos_weight": round(scale_pos_weight, 4),
        },
        "split": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "metrics": metrics,
        "confusion_matrix": confusion,
        "decision_threshold": round(float(threshold), 4),
        "feature_importance": importances,
        "artifact_paths": {
            "model": str(model_path),
            "encoders": str(encoders_path),
            "metrics": str(metrics_path),
            "calibrator": str(calibrator_path),
        },
        "hyperparameters": hyperparameters,
        "tuning": tuning_info,
        "calibration": calibration_info,
    }


async def train_model(
    db: AsyncSession,
    tune: bool = TUNE_BY_DEFAULT,
    n_trials: int = DEFAULT_N_TRIALS,
    tuning_timeout: int | None = TUNING_TIMEOUT_SECONDS,
) -> dict:
    """Build the dataset from the DB and train an XGBoost model.

    Parameters
    ----------
    db : AsyncSession
        Database session for querying claims.
    tune : bool
        Enable Optuna hyperparameter tuning (default ``True``).
    n_trials : int
        Maximum Optuna trials when tuning is enabled.
    tuning_timeout : int | None
        Wall-clock budget for tuning in seconds (``None`` for no limit).

    Returns
    -------
    dict
        Training results from ``_run_training``, augmented with the
        ``training_id`` of the persisted history record.
    """
    df = await build_dataset(db)

    if df.empty:
        raise ValueError("No labelled claims found in the database")

    # Offload the CPU-bound train (Optuna CV + XGBoost fits) to a worker
    # thread so it doesn't block the event loop for the tuning budget. The DB
    # session is untouched inside _run_training, so this is safe.
    result = await asyncio.to_thread(
        _run_training, df, tune, n_trials, tuning_timeout
    )
    training_id = await _persist_training_metrics(db, result)
    result["training_id"] = training_id
    return result


async def _persist_training_metrics(db: AsyncSession, result: dict) -> str:
    """Insert a row into ``model_training_metrics`` capturing this run.

    Why: lets the UI surface previous runs, dataset sizes, and accuracy
    trends without re-reading artifact files. The ``training_id`` returned
    here is also echoed in the trainer's response so the caller can link
    immediately to the history record.
    """
    stats = result.get("dataset_stats", {})
    metrics = result.get("metrics", {})
    split = result.get("split", {})

    trained_at_raw = result.get("trained_at")
    try:
        trained_at = (
            datetime.fromisoformat(trained_at_raw)
            if trained_at_raw
            else datetime.now(timezone.utc)
        )
    except ValueError:
        trained_at = datetime.now(timezone.utc)

    training_id = str(uuid.uuid4())

    record = TrainingMetric(
        training_id=training_id,
        training_timestamp=trained_at,
        total_claims_used=int(stats.get("total", 0)),
        training_samples=int(split.get("train_samples", 0)),
        test_samples=int(split.get("test_samples", 0)),
        denied_claims=int(stats.get("denied", 0)),
        paid_claims=int(stats.get("paid", 0)),
        denial_rate=float(stats.get("denial_rate", 0.0) or 0.0),
        accuracy=float(metrics.get("accuracy", 0.0) or 0.0),
        precision=float(metrics.get("precision", 0.0) or 0.0),
        recall=float(metrics.get("recall", 0.0) or 0.0),
        f1_score=float(metrics.get("f1", 0.0) or 0.0),
        roc_auc=(
            float(metrics["roc_auc"])
            if metrics.get("roc_auc") is not None
            else None
        ),
        training_time_seconds=float(result.get("training_time_seconds", 0.0)),
        model_version=MODEL_VERSION,
        notes=None,
        status=str(result.get("status", "success")),
    )

    db.add(record)
    await db.commit()
    await db.refresh(record)
    logger.info(
        "Persisted training metrics record id=%d training_id=%s",
        record.id,
        training_id,
    )
    return training_id


if __name__ == "__main__":
    import argparse
    import sys

    from app.core.database import async_session

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train the denial-prediction model")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip Optuna tuning and use baseline hyperparameters",
    )
    parser.add_argument(
        "--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Max Optuna trials"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TUNING_TIMEOUT_SECONDS,
        help="Tuning wall-clock budget in seconds",
    )
    args = parser.parse_args()

    async def _main() -> None:
        async with async_session() as db:
            results = await train_model(
                db,
                tune=not args.no_tune,
                n_trials=args.n_trials,
                tuning_timeout=args.timeout,
            )
            print(json.dumps(results, indent=2))

    try:
        asyncio.run(_main())
    except ValueError as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)
