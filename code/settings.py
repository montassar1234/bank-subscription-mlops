from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "code"
DATA_DIR = PROJECT_DIR / "data"

MLFLOW_DB_PATH = PROJECT_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = PROJECT_DIR / "mlartifacts"
MLFLOW_EXPERIMENT_NAME = "Bank_Marketing_Boosting"

MODEL_PATH = DATA_DIR / "best_model_pipeline.pkl"
METRICS_PATH = DATA_DIR / "model_metrics.csv"
RAW_DATA_PATH = DATA_DIR / "bank-full.csv"
CLEAN_DATA_PATH = DATA_DIR / "bank-full-clean.csv"
FEATURE_ANALYSIS_PATH = DATA_DIR / "feature_analysis_report.csv"
PREPARATION_SUMMARY_PATH = DATA_DIR / "data_preparation_summary.json"

TARGET_COLUMN = "y"
MODELING_EXCLUDED_FEATURES = ["duration"]
MODELING_FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]


def sqlite_uri(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve().as_posix()}"


def file_uri(path: Path) -> str:
    return path.resolve().as_uri()


DEFAULT_MLFLOW_BACKEND_URI = sqlite_uri(MLFLOW_DB_PATH)
DEFAULT_MLFLOW_ARTIFACT_ROOT = file_uri(MLFLOW_ARTIFACTS_DIR)
