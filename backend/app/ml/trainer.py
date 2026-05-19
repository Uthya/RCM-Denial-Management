"""XGBoost training pipeline for claim denial prediction.

Trains an XGBClassifier on the feature-engineered dataset, evaluates
performance, and persists the model + metrics as artifacts.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sqlalchemy.ext.asyncio import AsyncSession
from xgboost import XGBClassifier

from app.core.config import settings
from app.ml.dataset import build_dataset, get_dataset_stats
from app.ml.feature_engineering import FEATURE_COLUMNS, FeatureEngineer

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TRAINING_SAMPLES = 50


def _run_training(df: pd.DataFrame) -> dict:
    """Train an XGBClassifier and return results with metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw labelled DataFrame from ``build_dataset()``.

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

    # Feature engineering
    engineer = FeatureEngineer()
    X = engineer.fit_transform(df)

    # Extract label from original df (feature engineer drops it as leakage)
    y = df["denied"]

    # Compute class balance weight
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    logger.info(
        "Class balance — positive: %d, negative: %d, scale_pos_weight: %.3f",
        n_positive,
        n_negative,
        scale_pos_weight,
    )

    # Train/test split (stratify only when both classes are present)
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    # Hyperparameters
    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "scale_pos_weight": round(scale_pos_weight, 4),
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
    }

    # Train
    model = XGBClassifier(
        n_estimators=hyperparameters["n_estimators"],
        max_depth=hyperparameters["max_depth"],
        learning_rate=hyperparameters["learning_rate"],
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

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
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
        except ValueError:
            metrics["roc_auc"] = None
            logger.warning("Could not compute ROC-AUC")
    else:
        metrics["roc_auc"] = None
        logger.warning("Skipping ROC-AUC: only one class in test set")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    confusion = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    logger.info("Test metrics: %s", metrics)
    logger.info("Confusion matrix: %s", confusion)

    # Feature importance
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

    trained_at = datetime.now(timezone.utc).isoformat()
    training_time = round(time.time() - start_time, 3)

    # Write training metrics JSON
    metrics_path = Path(settings.ML_METRICS_PATH)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_artifact = {
        "metrics": metrics,
        "feature_importance": importances,
        "hyperparameters": hyperparameters,
        "dataset_size": len(df),
        "trained_at": trained_at,
    }
    metrics_path.write_text(json.dumps(metrics_artifact, indent=2))
    logger.info("Saved training metrics to %s", metrics_path)

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
        "feature_importance": importances,
        "artifact_paths": {
            "model": str(model_path),
            "encoders": str(encoders_path),
            "metrics": str(metrics_path),
        },
        "hyperparameters": hyperparameters,
    }


async def train_model(db: AsyncSession) -> dict:
    """Build the dataset from the DB and train an XGBoost model.

    Parameters
    ----------
    db : AsyncSession
        Database session for querying claims.

    Returns
    -------
    dict
        Training results from ``_run_training``.
    """
    df = await build_dataset(db)

    if df.empty:
        raise ValueError("No labelled claims found in the database")

    return _run_training(df)


if __name__ == "__main__":
    import asyncio
    import sys

    from app.core.database import async_session

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    async def _main() -> None:
        async with async_session() as db:
            results = await train_model(db)
            print(json.dumps(results, indent=2))

    try:
        asyncio.run(_main())
    except ValueError as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)
