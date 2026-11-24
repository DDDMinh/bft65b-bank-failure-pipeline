"""Train-only sklearn preprocessing for the frozen Phase 4 feature matrix."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bft65b.features.registry import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from bft65b.features.spec import FeatureError, resolve_feature_path
from bft65b.features.validation import validate_preprocessing_fit_frame
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.io import atomic_write_json, sha256_file


class TrainQuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric columns to quantiles learned only during ``fit``.

    All-null Dev columns receive unbounded bounds and are retained. This keeps
    the output width stable when expanding CV folds begin with all-null fields.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: Any, y: Any = None) -> "TrainQuantileClipper":
        values = np.asarray(X, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lower = np.nanquantile(values, self.lower_quantile, axis=0)
            upper = np.nanquantile(values, self.upper_quantile, axis=0)
        self.lower_ = np.where(np.isfinite(lower), lower, -np.inf)
        self.upper_ = np.where(np.isfinite(upper), upper, np.inf)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[1] != self.n_features_in_:
            raise FeatureError(
                f"Clipper expects {self.n_features_in_} columns; received {values.shape[1]}"
            )
        return np.clip(values, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        if input_features is None:
            return np.asarray(
                [f"clipped_{index}" for index in range(self.n_features_in_)], dtype=object
            )
        return np.asarray(input_features, dtype=object)


class TrainColumnImputer(BaseEstimator, TransformerMixin):
    """Column-wise imputer with train-only statistics and bounded memory.

    sklearn's dense median imputer materialises a full array of ``argsort``
    indices, which exceeds laptop memory for the 1.26M x 102 full-Dev fit.
    Computing one column statistic at a time is mathematically equivalent for
    median/mean imputation while keeping peak extra memory to one column.
    """

    def __init__(
        self,
        strategy: str = "median",
        *,
        add_indicator: bool = False,
        keep_empty_features: bool = True,
        fill_value: float = 0.0,
    ):
        self.strategy = strategy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.fill_value = fill_value

    def fit(self, X: Any, y: Any = None) -> "TrainColumnImputer":
        values = np.asarray(X, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.n_features_in_ = values.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if self.strategy == "median":
                statistics = np.array(
                    [np.nanmedian(values[:, index]) for index in range(values.shape[1])]
                )
            elif self.strategy == "mean":
                statistics = np.array(
                    [np.nanmean(values[:, index]) for index in range(values.shape[1])]
                )
            elif self.strategy == "constant":
                statistics = np.full(values.shape[1], float(self.fill_value))
            else:
                raise FeatureError(f"Unsupported imputation strategy: {self.strategy}")
        self.statistics_ = np.where(np.isfinite(statistics), statistics, float(self.fill_value))
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[1] != self.n_features_in_:
            raise FeatureError(
                f"Imputer expects {self.n_features_in_} columns; received {values.shape[1]}"
            )
        result = np.array(values, dtype=np.float64, copy=True)
        for index, statistic in enumerate(self.statistics_):
            column = result[:, index]
            missing = np.isnan(column)
            if missing.any():
                column[missing] = statistic
        return result

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        if input_features is None:
            return np.asarray(
                [f"imputed_{index}" for index in range(self.n_features_in_)], dtype=object
            )
        return np.asarray(input_features, dtype=object)


def build_preprocessor(
    config: Mapping[str, Any],
    *,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Create an unfitted preprocessing Pipeline.

    Phase 5 must call this factory and fit the returned object only on its own
    training fold. No global statistics are read or cached here.
    """
    numeric = list(numeric_features)
    categorical = list(categorical_features)
    numeric_pipeline = Pipeline(
        steps=[
            (
                "clip",
                TrainQuantileClipper(
                    lower_quantile=float(config["features"]["clipping_lower_quantile"]),
                    upper_quantile=float(config["features"]["clipping_upper_quantile"]),
                ),
            ),
            (
                "impute",
                TrainColumnImputer(
                    strategy=str(config["features"]["imputation_strategy"]),
                    add_indicator=False,
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler(copy=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_dev_preprocessor(
    config: Mapping[str, Any],
    feature_frame: Any,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Fit and persist preprocessing on development rows only."""
    if not hasattr(feature_frame, "to_pandas"):
        raise FeatureError("fit_dev_preprocessor expects a Polars DataFrame")
    guard = validate_preprocessing_fit_frame(feature_frame, config)
    pandas_frame = feature_frame.to_pandas()
    missing_features = [
        name
        for name in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
        if name not in pandas_frame.columns
    ]
    if missing_features:
        raise FeatureError(f"Feature frame is missing features: {missing_features}")

    dev_numeric = pandas_frame.loc[:, list(NUMERIC_FEATURES)]
    all_null = [
        name for name in NUMERIC_FEATURES if dev_numeric[name].isna().all()
    ]
    coverage = {
        name: round(100.0 * float(dev_numeric[name].notna().mean()), 6)
        for name in NUMERIC_FEATURES
    }

    preprocessor = build_preprocessor(config)
    preprocessor.fit(
        pandas_frame.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
    )
    transformed_width = preprocessor.transform(
        pandas_frame.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]].head(2)
    ).shape[1]

    path = resolve_feature_path(config, "preprocessor_path", root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)

    manifest = {
        "schema_version": 1,
        "fitted_on": "development rows only",
        "fit_guard": guard,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "metadata_columns_excluded": [
            "CERT", "REPDTE", "qidx", "year", "quarter", "split"
        ],
        "target_column_excluded": "y_4q",
        "zero_dev_coverage_features": sorted(
            name for name, value in coverage.items() if value == 0.0
        ),
        "all_null_dev_features": sorted(all_null),
        "dev_coverage_pct": coverage,
        "settings": {
            "clipping_lower_quantile": float(config["features"]["clipping_lower_quantile"]),
            "clipping_upper_quantile": float(config["features"]["clipping_upper_quantile"]),
            "imputation_strategy": str(config["features"]["imputation_strategy"]),
            "imputer_add_indicator": False,
            "imputer_keep_empty_features": True,
            "scaler": "StandardScaler",
            "categorical_imputation": "constant:missing",
            "categorical_encoder": "OneHotEncoder(handle_unknown='ignore')",
        },
        "width": {
            "input_features": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
            "transformed_columns": int(transformed_width),
        },
        "artifact_path": str(path),
        "artifact_sha256": sha256_file(path),
    }
    manifest_path = resolve_feature_path(config, "preprocessing_fit_path", root=root)
    atomic_write_json(manifest_path, manifest)
    return manifest
