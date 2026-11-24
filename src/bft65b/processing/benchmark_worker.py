"""Single-run benchmark worker.

The orchestrator launches one fresh process per engine/run so that Polars,
pandas, and Spark cannot reuse another engine's in-process caches or an old
SparkSession. This module executes exactly one standardized workload run and
emits a machine-readable result line on stdout.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Sequence

from bft65b.config import load_config
from bft65b.paths import PROJECT_ROOT
from bft65b.processing.pandas_engine import run_pandas
from bft65b.processing.polars_engine import run_polars
from bft65b.processing.spark_engine import run_spark
from bft65b.processing.spec import (
    benchmark_output_path,
    reset_output_path,
    scale_spec,
)

RESULT_MARKER = "BENCHMARK_RESULT_JSON:"
RUNNERS = {"polars": run_polars, "pandas": run_pandas, "spark": run_spark}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=sorted(RUNNERS), required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    spec = scale_spec(config, args.scale, root=PROJECT_ROOT)
    output_path = benchmark_output_path(
        config, args.engine, args.scale, args.run, root=PROJECT_ROOT
    )
    reset_output_path(output_path)

    started = time.perf_counter()
    result = RUNNERS[args.engine](spec, config, output_path, root=PROJECT_ROOT)
    elapsed = time.perf_counter() - started

    payload = {
        "engine": args.engine,
        "scale_percent": args.scale,
        "run": args.run,
        "runtime_seconds": elapsed,
        "rows": result["rows"],
        "columns": result["columns"],
        "output_path": str(output_path),
    }
    print(RESULT_MARKER + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
