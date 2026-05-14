from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from data_preparation import build_preprocessor, prepare_datasets
from mlflow_utils import configure_mlflow, get_artifact_root
from settings import (
    CLEAN_DATA_PATH,
    DATA_DIR,
    FEATURE_ANALYSIS_PATH,
    METRICS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_PATH,
    MODELING_EXCLUDED_FEATURES,
    MODELING_FEATURE_COLUMNS,
    PREPARATION_SUMMARY_PATH,
    TARGET_COLUMN,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

def load_dataset(file_path: Path = CLEAN_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(file_path)
    X = df[MODELING_FEATURE_COLUMNS].copy()
    y = (df[TARGET_COLUMN] == "yes").astype(int)
    return X, y


def model_configs() -> dict:
    configs = {
        "LogisticRegression_Baseline": {
            "model": LogisticRegression(random_state=42, class_weight="balanced", max_iter=1500),
            "use_smote": False,
            "params": {
                "classifier__C": [0.5, 1.0, 2.0],
            },
        },
        "GradientBoosting_SMOTE": {
            "model": GradientBoostingClassifier(random_state=42),
            "use_smote": True,
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2, 3],
            },
        },
        "RandomForest_Baseline": {
            "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "use_smote": False,
            "params": {
                "classifier__n_estimators": [150],
                "classifier__max_depth": [12, None],
            },
        },
    }

    if XGBClassifier is not None:
        configs["XGBoost_SMOTE"] = {
            "model": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
            ),
            "use_smote": True,
            "params": {
                "classifier__n_estimators": [150, 250],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
            },
        }
    return configs


def save_confusion_matrix_png(y_true: pd.Series, y_pred, model_name: str) -> Path:
    out = DATA_DIR / f"confusion_matrix_{model_name}.png"
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def save_roc_curve_png(y_true: pd.Series, y_prob, model_name: str) -> Path:
    out = DATA_DIR / f"roc_curve_{model_name}.png"
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def ensure_experiment(name: str) -> str:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id

    artifact_root = get_artifact_root().rstrip("/")
    experiment_id = client.create_experiment(
        name=name,
        artifact_location=f"{artifact_root}/{name}",
    )
    return experiment_id


def run_experiment(
    name: str,
    model,
    params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    use_smote: bool,
) -> tuple[ImbPipeline, dict]:
    pipeline_steps = [("preprocessor", preprocessor)]
    if use_smote:
        pipeline_steps.append(("smote", SMOTE(random_state=42)))
    pipeline_steps.append(("classifier", model))
    pipeline = ImbPipeline(steps=pipeline_steps)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"\n{name} best params: {search.best_params_}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    cm_path = save_confusion_matrix_png(y_test, y_pred, name)
    roc_path = save_roc_curve_png(y_test, y_prob, name)

    mlflow.log_params(search.best_params_)
    mlflow.log_metrics(
        {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        }
    )
    mlflow.log_artifact(str(cm_path))
    mlflow.log_artifact(str(roc_path))
    mlflow.sklearn.log_model(best_model, artifact_path="model")
    return best_model, metrics


def train_and_select_best_model() -> None:
    tracking_uri = configure_mlflow()
    prepare_datasets()
    X, y = load_dataset()
    preprocessor = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    experiment_id = ensure_experiment(MLFLOW_EXPERIMENT_NAME)
    all_metrics = []
    best_f1 = -1.0
    best_model = None

    for name, cfg in model_configs().items():
        with mlflow.start_run(experiment_id=experiment_id, run_name=name):
            mlflow.log_param("feature_set", "pre_contact")
            mlflow.log_param(
                "excluded_features",
                ",".join(MODELING_EXCLUDED_FEATURES) if MODELING_EXCLUDED_FEATURES else "none",
            )
            mlflow.log_param("feature_count", len(MODELING_FEATURE_COLUMNS))
            mlflow.log_param("preparation_summary_path", str(PREPARATION_SUMMARY_PATH))
            mlflow.log_param("feature_analysis_path", str(FEATURE_ANALYSIS_PATH))
            model, metrics = run_experiment(
                name=name,
                model=cfg["model"],
                params=cfg["params"],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                preprocessor=preprocessor,
                use_smote=cfg.get("use_smote", False),
            )
            all_metrics.append(metrics)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model = model

    if best_model is None:
        raise RuntimeError("No model trained successfully.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    pd.DataFrame(all_metrics).sort_values("f1", ascending=False).to_csv(
        METRICS_PATH, index=False
    )
    print(f"\nBest model saved to: {MODEL_PATH}")
    print(f"Model metrics saved to: {METRICS_PATH}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")


if __name__ == "__main__":
    train_and_select_best_model()
