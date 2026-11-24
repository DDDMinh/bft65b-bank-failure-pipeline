"""Fixed sklearn estimators wrapped with the train-only Phase 4 preprocessor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from bft65b.features.preprocessing import build_preprocessor
from bft65b.features.registry import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from bft65b.modelling.spec import (
    CORE_BASELINE_FEATURES,
    ModelContractError,
    validate_feature_contract,
)
from bft65b.utils.io import sha256_file


def features_for_model(model_name: str) -> list[str]:
    """Return the frozen input columns for one sklearn model."""
    if model_name == "baseline_logistic":
        validate_feature_contract(CORE_BASELINE_FEATURES, target_present=True)
        return list(CORE_BASELINE_FEATURES)
    if model_name == "primary_hgb":
        return list(FEATURE_COLUMNS)
    raise ModelContractError(f"Unknown sklearn model: {model_name}")


def build_model_pipeline(
    config: Mapping[str, Any],
    model_name: str,
    weighting_mode: str,
    *,
    seed: int,
) -> Pipeline:
    """Create an unfitted Pipeline; no persisted preprocessing state is loaded."""
    if weighting_mode not in set(config["model"]["candidate_weighting_modes"]):
        raise ModelContractError(f"Unsupported weighting mode: {weighting_mode}")
    feature_columns = features_for_model(model_name)
    if model_name == "baseline_logistic":
        numeric = list(feature_columns)
        categorical: list[str] = []
        classifier = LogisticRegression(
            C=float(config["model"]["baseline"]["C"]),
            max_iter=int(config["model"]["baseline"]["max_iter"]),
            solver="lbfgs",
            class_weight="balanced" if weighting_mode == "balanced" else None,
            random_state=int(seed),
        )
    elif model_name == "primary_hgb":
        numeric = list(NUMERIC_FEATURES)
        categorical = list(CATEGORICAL_FEATURES)
        settings = config["model"]["hgb"]
        classifier = HistGradientBoostingClassifier(
            learning_rate=float(settings["learning_rate"]),
            max_iter=int(settings["max_iter"]),
            max_leaf_nodes=int(settings["max_leaf_nodes"]),
            min_samples_leaf=int(settings["min_samples_leaf"]),
            l2_regularization=float(settings["l2_regularization"]),
            random_state=int(seed),
        )
    else:
        raise ModelContractError(f"Unknown sklearn model: {model_name}")

    preprocessor = build_preprocessor(
        config, numeric_features=numeric, categorical_features=categorical
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def training_sample_weight(
    y: np.ndarray,
    model_name: str,
    weighting_mode: str,
) -> np.ndarray | None:
    """Compute class weights from the training fold only."""
    if weighting_mode != "balanced" or model_name != "primary_hgb":
        return None
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        raise ModelContractError("Balanced HGB requires at least one training positive")
    return np.where(y == 1, 1.0, negatives / positives).astype(np.float64)


def fit_model(
    X: pd.DataFrame,
    y: Sequence[int] | np.ndarray,
    config: Mapping[str, Any],
    model_name: str,
    weighting_mode: str,
    *,
    seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit one frozen model variant on the supplied training rows."""
    feature_columns = features_for_model(model_name)
    missing = [name for name in feature_columns if name not in X.columns]
    if missing:
        raise ModelContractError(f"Training frame is missing features: {missing}")
    target = np.asarray(y, dtype=np.int8)
    if target.size != len(X) or np.unique(target).size < 2:
        raise ModelContractError("Model fit requires a non-empty two-class training target")
    pipeline = build_model_pipeline(config, model_name, weighting_mode, seed=seed)
    weights = training_sample_weight(target, model_name, weighting_mode)
    fit_kwargs: dict[str, Any] = {}
    if weights is not None:
        fit_kwargs["classifier__sample_weight"] = weights
    pipeline.fit(X.loc[:, feature_columns], target, **fit_kwargs)
    metadata = {
        "model_name": model_name,
        "weighting_mode": weighting_mode,
        "seed": int(seed),
        "features": feature_columns,
        "training_rows": int(len(X)),
        "training_positive_rows": int(target.sum()),
        "class_weight": "balanced" if weighting_mode == "balanced" else None,
        "calibrator": "none; unweighted probabilities selected by Dev OOF Brier where applicable",
    }
    return pipeline, metadata


def predict_probability(pipeline: Pipeline, X: pd.DataFrame, model_name: str) -> np.ndarray:
    """Return the positive-class probability vector."""
    features = features_for_model(model_name)
    missing = [name for name in features if name not in X.columns]
    if missing:
        raise ModelContractError(f"Prediction frame is missing features: {missing}")
    probabilities = pipeline.predict_proba(X.loc[:, features])
    return np.asarray(probabilities[:, 1], dtype=np.float64)


def save_model_artifact(
    pipeline: Pipeline,
    metadata: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    """Persist a fitted model with immutable metadata and return its hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": dict(metadata)}, path)
    return {"path": str(path), "sha256": sha256_file(path)}


def load_model_artifact(path: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load a frozen model artifact and verify its structure."""
    if not path.exists():
        raise ModelContractError(f"Frozen model artifact not found: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "pipeline" not in payload or "metadata" not in payload:
        raise ModelContractError(f"Invalid model artifact: {path}")
    return payload["pipeline"], dict(payload["metadata"])
