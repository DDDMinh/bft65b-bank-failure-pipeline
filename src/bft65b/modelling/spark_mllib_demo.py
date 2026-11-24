"""Spark MLlib parity demo on the frozen Dev split.

The stage order is deliberately::

    Imputer -> VectorAssembler -> StandardScaler -> LogisticRegression

Spark's ``Imputer`` accepts scalar numeric columns, so it must run before
``VectorAssembler``. Columns that are entirely null in Dev are removed before
the Imputer rather than being fed to it. No Test row is read here; Test parity
is computed only inside the single guarded ``evaluate-test`` run.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bft65b.features.registry import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from bft65b.modelling.metrics import binary_metrics
from bft65b.modelling.models import load_model_artifact, predict_probability
from bft65b.modelling.spec import ModelContractError, resolve_model_path
from bft65b.paths import PROJECT_ROOT
from bft65b.utils.io import atomic_write_json, sha256_file


def source_column(name: str) -> str:
    return f"num__{name}"


def imputed_column(name: str) -> str:
    return f"imp__{name}"


def category_column(feature: str, index: int) -> str:
    return f"cat__{feature}__{index}"


SPARK_STAGE_ORDER: tuple[str, ...] = (
    "Imputer",
    "VectorAssembler",
    "StandardScaler",
    "LogisticRegression",
)


def compute_spark_contract(
    dev_frame: pd.DataFrame, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive the Spark feature contract from Dev rows only."""
    missing = [name for name in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES) if name not in dev_frame]
    if missing:
        raise ModelContractError(f"Spark contract is missing Dev columns: {missing}")
    lower_q = float(config["features"]["clipping_lower_quantile"])
    upper_q = float(config["features"]["clipping_upper_quantile"])
    numeric_columns: list[str] = []
    bounds: dict[str, list[float]] = {}
    all_null: list[str] = []
    for name in NUMERIC_FEATURES:
        values = pd.to_numeric(dev_frame[name], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            all_null.append(name)
            continue
        lower, upper = np.quantile(finite, [lower_q, upper_q])
        if not np.isfinite(lower) or not np.isfinite(upper):
            all_null.append(name)
            continue
        if lower == upper:
            spread = max(abs(lower) * 1e-6, 1e-6)
            lower -= spread
            upper += spread
        numeric_columns.append(name)
        bounds[name] = [float(lower), float(upper)]

    categories: dict[str, list[str]] = {}
    for name in CATEGORICAL_FEATURES:
        values = sorted(
            str(value) for value in dev_frame[name].dropna().unique().tolist() if str(value)
        )
        categories[name] = values
    categorical_columns = [
        category_column(name, index) for name, values in categories.items() for index in range(len(values))
    ]
    if not numeric_columns and not categorical_columns:
        raise ModelContractError("Spark contract has no usable Dev features")
    return {
        "numeric_columns": numeric_columns,
        "numeric_bounds": bounds,
        "all_null_dev_columns_dropped": all_null,
        "categorical_values": categories,
        "categorical_columns": categorical_columns,
        "source_columns": [source_column(name) for name in numeric_columns],
        "imputed_columns": [imputed_column(name) for name in numeric_columns],
        "imputer_strategy": "median",
        "clipping_lower_quantile": lower_q,
        "clipping_upper_quantile": upper_q,
        "scaler": "StandardScaler(withMean=True, withStd=True)",
        "classifier": "LogisticRegression(standardization=False, unweighted)",
    }


def build_spark_stages(
    contract: Mapping[str, Any], config: Mapping[str, Any]
) -> list[Any]:
    """Create the frozen Spark ML stages in the mandated order."""
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler

    imputer = Imputer(
        inputCols=list(contract["source_columns"]),
        outputCols=list(contract["imputed_columns"]),
        strategy=str(contract["imputer_strategy"]),
    )
    assembler = VectorAssembler(
        inputCols=[*contract["imputed_columns"], *contract["categorical_columns"]],
        outputCol="features",
        handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=True,
    )
    settings = config["model"]["spark"]
    classifier = LogisticRegression(
        featuresCol="scaled_features",
        labelCol="y_4q",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="raw_prediction",
        maxIter=int(settings["max_iter"]),
        regParam=float(settings["reg_param"]),
        elasticNetParam=0.0,
        standardization=False,
    )
    return Pipeline(stages=[imputer, assembler, scaler, classifier]).getStages()


def stage_names(contract: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Return the frozen stage order without requiring an active SparkContext."""
    return list(SPARK_STAGE_ORDER)


def create_spark_session(config: Mapping[str, Any]):
    """Create a local SparkSession with Arrow enabled for pandas interchange."""
    from bft65b.processing.spark_session import create_local_spark_session

    spark = create_local_spark_session(
        "bft65b-spark-mllib-demo",
        master="local[2]",
        shuffle_partitions=int(config["benchmark"]["spark_shuffle_partitions"]),
        driver_memory=str(config["benchmark"]["spark_driver_memory"]),
    )
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def prepare_spark_frame(
    spark: Any,
    pandas_frame: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    with_label: bool,
):
    """Apply Dev-frozen clipping/one-hot encoding and convert NaN to Spark null."""
    from pyspark.sql.functions import col, isnan, lit, when

    prepared = pd.DataFrame()
    prepared["_row_id"] = np.arange(len(pandas_frame), dtype=np.int64)
    for metadata in ("CERT", "qidx", "REPDTE"):
        if metadata in pandas_frame.columns:
            prepared[metadata] = pandas_frame[metadata].to_numpy()
    if with_label:
        prepared["y_4q"] = pandas_frame["y_4q"].astype(np.int8).to_numpy()

    bounds = contract["numeric_bounds"]
    for name in contract["numeric_columns"]:
        lower, upper = bounds[name]
        values = pd.to_numeric(pandas_frame[name], errors="coerce").to_numpy(dtype=np.float64)
        prepared[source_column(name)] = np.clip(values, lower, upper)

    for name, values in contract["categorical_values"].items():
        observed = pandas_frame[name].astype("object")
        for index, value in enumerate(values):
            prepared[category_column(name, index)] = (observed == value).astype(np.int8).to_numpy()

    frame = spark.createDataFrame(prepared)
    for name in contract["source_columns"]:
        frame = frame.withColumn(
            name,
            when(isnan(col(name)) | col(name).isNull(), lit(None).cast("double")).otherwise(
                col(name).cast("double")
            ),
        )
    return frame


def train_spark_model(
    config: Mapping[str, Any],
    dev_pandas: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Fit the frozen Spark MLlib model on Dev and persist it."""
    from pyspark.ml import Pipeline

    spark = create_spark_session(config)
    try:
        frame = prepare_spark_frame(spark, dev_pandas, contract, with_label=True)
        stages = build_spark_stages(contract, config)
        model = Pipeline(stages=stages).fit(frame)
        path = resolve_model_path(config, "spark_model_path", root=root)
        if path.exists():
            shutil.rmtree(path)
        model.save(path.as_posix())
        try:
            sample_plan = (
                model.transform(frame.limit(20))._jdf.queryExecution().executedPlan().toString()
            )
        except Exception:  # plan capture is diagnostic, not part of the model contract
            sample_plan = "Spark executed plan unavailable through the installed PySpark API"
        return {
            "artifact_path": str(path),
            "artifact_sha256": sha256_directory(path),
            "training_rows": int(frame.count()),
            "executed_plan": sample_plan,
        }
    finally:
        spark.stop()


def sha256_directory(path: Path) -> str:
    """Stable hash of every file in a saved Spark model directory."""
    import hashlib

    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(child.read_bytes())
    return digest.hexdigest()


def score_spark_model(
    config: Mapping[str, Any],
    pandas_frame: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Score a frame with the frozen Spark model and return probabilities in input order."""
    from pyspark.ml.pipeline import PipelineModel
    from pyspark.sql.functions import col, udf
    from pyspark.sql.types import DoubleType

    model_path = resolve_model_path(config, "spark_model_path", root=root)
    if not model_path.exists():
        raise ModelContractError(f"Spark model artifact not found: {model_path}")
    spark = create_spark_session(config)
    try:
        frame = prepare_spark_frame(spark, pandas_frame, contract, with_label=False)
        model = PipelineModel.load(model_path.as_posix())
        scored = model.transform(frame)
        positive_probability = udf(
            lambda vector: float(vector[1]) if vector is not None else float("nan"),
            DoubleType(),
        )
        collected = (
            scored.select("_row_id", positive_probability(col("probability")).alias("probability"))
            .orderBy("_row_id")
            .toPandas()
        )
        result = pandas_frame.loc[:, [c for c in ("CERT", "REPDTE", "qidx") if c in pandas_frame.columns]].copy()
        result["probability"] = collected["probability"].to_numpy(dtype=np.float64)
        return result
    finally:
        spark.stop()


def run_spark_demo(
    config: Mapping[str, Any],
    *,
    split: str = "dev",
    force: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Train or verify the Spark demo and write a Dev-only manifest."""
    from bft65b.modelling.cv import load_dev_frame
    from bft65b.modelling.spec import read_json, verify_frozen_artifacts

    if split != "dev":
        raise ModelContractError("Spark demo supports only the Dev split; Test is guarded")
    dev = load_dev_frame(config, root=root)
    dev_pandas = dev.to_pandas()
    model_path = resolve_model_path(config, "spark_model_path", root=root)
    spec_path = resolve_model_path(config, "frozen_spec_path", root=root)
    frozen = read_json(spec_path) if spec_path.exists() else None

    if frozen is not None:
        if force:
            raise ModelContractError("Refusing Spark --force after the model specification is frozen")
        verify_frozen_artifacts(frozen, config, root=root)
        manifest_contract = frozen["spark_contract"]
        training = {
            "artifact_path": str(model_path),
            "artifact_sha256": frozen["model_artifacts"]["spark_logistic"]["sha256"],
            "training_rows": frozen.get("modelling_inputs", {}).get("dev_rows"),
            "executed_plan": "Loaded verified frozen Spark PipelineModel; no refit performed",
        }
        threshold = float(frozen["model_artifacts"]["spark_logistic"]["threshold"])
    elif force or not model_path.exists():
        contract = compute_spark_contract(dev_pandas, config)
        training = train_spark_model(config, dev_pandas, contract, root=root)
        manifest_contract = contract
        threshold = 0.5
    else:
        raise ModelContractError("Spark model exists without a frozen specification")

    scored = score_spark_model(config, dev_pandas, manifest_contract, root=root)
    scored["y_4q"] = dev_pandas["y_4q"].to_numpy(dtype=np.int8)
    spark_metrics = binary_metrics(scored["y_4q"], scored["probability"], float(threshold))

    comparison: dict[str, Any] = {}
    baseline_path = resolve_model_path(config, "baseline_artifact_path", root=root)
    if baseline_path.exists():
        baseline_pipeline, metadata = load_model_artifact(baseline_path)
        sample = dev_pandas.sample(n=min(50_000, len(dev_pandas)), random_state=int(config["project"]["random_seed"]))
        baseline_probability = predict_probability(baseline_pipeline, sample, "baseline_logistic")
        spark_sample = scored.loc[sample.index, "probability"].to_numpy(dtype=np.float64)
        correlation = spearmanr(baseline_probability, spark_sample).statistic
        comparison = {
            "sample_rows": int(len(sample)),
            "baseline_model": metadata.get("weighting_mode"),
            "spearman_rank_correlation": float(correlation),
            "note": "In-sample Dev demo; not a validation performance estimate",
        }

    manifest = {
        "schema_version": 1,
        "split": "dev",
        "stage_order": stage_names(manifest_contract, config),
        "contract": manifest_contract,
        "training": training,
        "threshold": float(threshold),
        "threshold_source": (
            "frozen_spark_model_artifact"
            if frozen is not None
            else "pre_freeze_default_0.5"
        ),
        "dev_in_sample_metrics": spark_metrics,
        "baseline_comparison": comparison,
    }
    manifest_path = resolve_model_path(config, "spark_demo_manifest_path", root=root)
    atomic_write_json(manifest_path, manifest)
    return manifest
