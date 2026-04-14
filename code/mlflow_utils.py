import os
from pathlib import Path

import mlflow

from settings import (
    DEFAULT_MLFLOW_ARTIFACT_ROOT,
    DEFAULT_MLFLOW_BACKEND_URI,
    MLFLOW_ARTIFACTS_DIR,
)


def ensure_runtime_directories() -> None:
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def configure_mlflow() -> str:
    ensure_runtime_directories()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_BACKEND_URI)
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def get_artifact_root() -> str:
    return os.getenv("MLFLOW_ARTIFACT_ROOT", DEFAULT_MLFLOW_ARTIFACT_ROOT)


def describe_mlflow_runtime() -> dict[str, str]:
    return {
        "tracking_uri": mlflow.get_tracking_uri(),
        "artifact_root": get_artifact_root(),
        "artifacts_dir": str(Path(MLFLOW_ARTIFACTS_DIR).resolve()),
    }
