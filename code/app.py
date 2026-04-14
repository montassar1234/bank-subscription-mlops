from io import BytesIO
import json
from statistics import mean

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mlflow_utils import configure_mlflow, describe_mlflow_runtime
from settings import (
    CLEAN_DATA_PATH,
    FEATURE_ANALYSIS_PATH,
    METRICS_PATH,
    MODELING_EXCLUDED_FEATURES,
    MODELING_FEATURE_COLUMNS,
    MODEL_PATH,
    PREPARATION_SUMMARY_PATH,
)

FEATURE_COLUMNS = MODELING_FEATURE_COLUMNS

FEATURE_OPTIONS = {
    "job": [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ],
    "marital": ["single", "married", "divorced"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["yes", "no"],
    "housing": ["yes", "no"],
    "loan": ["yes", "no"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["unknown", "other", "failure", "success"],
}

DEFAULT_FORM = {
    "age": 35,
    "job": "technician",
    "marital": "single",
    "education": "secondary",
    "default": "no",
    "balance": 1200,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}

NUMERIC_FIELDS = ["age", "balance", "day", "campaign", "pdays", "previous"]

app = FastAPI(
    title="Bank Marketing Subscription API",
    description="Production-style scoring API for bank term-deposit subscription prediction.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

configure_mlflow()


def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        return pd.DataFrame(columns=["model", "accuracy", "f1", "roc_auc"])
    return pd.read_csv(METRICS_PATH)


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(CLEAN_DATA_PATH) if CLEAN_DATA_PATH.exists() else pd.DataFrame()


def load_feature_analysis() -> pd.DataFrame:
    if not FEATURE_ANALYSIS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_ANALYSIS_PATH)


def load_preparation_summary() -> dict:
    if not PREPARATION_SUMMARY_PATH.exists():
        return {}
    return json.loads(PREPARATION_SUMMARY_PATH.read_text(encoding="utf-8"))


def _format_percent(value: float) -> float:
    return round(value * 100, 2)


def _risk_band(probability_yes: float) -> str:
    if probability_yes >= 0.7:
        return "High propensity"
    if probability_yes >= 0.4:
        return "Moderate propensity"
    return "Low propensity"


def _recommended_action(probability_yes: float) -> str:
    if probability_yes >= 0.7:
        return "Prioritize this client for immediate personalized outreach."
    if probability_yes >= 0.4:
        return "Nurture with a targeted offer and follow-up campaign."
    return "Deprioritize direct sales contact and use lower-cost channels."


def _classifier_name() -> str:
    if model is None:
        return "Unavailable"
    classifier = model.named_steps.get("classifier")
    return classifier.__class__.__name__


def _get_feature_importance_records(top_n: int = 12) -> list[dict]:
    if model is None:
        return []

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out().tolist()

        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_.tolist()
        elif hasattr(classifier, "coef_"):
            coefficients = classifier.coef_
            importances = (
                coefficients[0].tolist()
                if hasattr(coefficients[0], "tolist")
                else list(coefficients[0])
            )
        else:
            return []

        ranked = sorted(
            [
                {
                    "feature": name.replace("num__", "").replace("cat__", ""),
                    "importance": round(abs(float(score)), 6),
                }
                for name, score in zip(feature_names, importances)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )
        return ranked[:top_n]
    except Exception:
        return []


def _scenario_delta_summary(base_payload: dict, scenario_payload: dict) -> list[str]:
    changed = []
    for key in FEATURE_COLUMNS:
        if base_payload.get(key) != scenario_payload.get(key):
            changed.append(
                f"{normalize_feature_name(key)}: {base_payload.get(key)} -> {scenario_payload.get(key)}"
            )
    return changed


def normalize_feature_name(name: str) -> str:
    return name.replace("_", " ").title()


def _strategic_recommendations(profile: dict) -> list[dict]:
    candidate_scenarios = [
        ("Upgrade contact channel", {**profile, "contact": "cellular"}),
        ("Lower campaign pressure", {**profile, "campaign": 1}),
        ("Improve account balance", {**profile, "balance": max(int(profile["balance"]), 2500)}),
        ("Move campaign to March", {**profile, "month": "mar"}),
        ("Use positive prior outcome", {**profile, "poutcome": "success"}),
    ]

    baseline_frame = pd.DataFrame([profile])
    _, baseline_probabilities = _scoring_result(baseline_frame)
    baseline_probability = float(baseline_probabilities[0])

    recommendations = []
    for title, scenario in candidate_scenarios:
        scenario_frame = pd.DataFrame([scenario])
        _, scenario_probabilities = _scoring_result(scenario_frame)
        scenario_probability = float(scenario_probabilities[0])
        recommendations.append(
            {
                "title": title,
                "baseline_probability_yes": round(baseline_probability, 4),
                "scenario_probability_yes": round(scenario_probability, 4),
                "uplift": round(scenario_probability - baseline_probability, 4),
            }
        )

    return sorted(recommendations, key=lambda item: item["uplift"], reverse=True)


def _dataset_overview(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "records": 0,
            "features": len(FEATURE_COLUMNS),
            "subscription_rate": 0.0,
            "avg_age": 0.0,
            "median_balance": 0.0,
            "campaign_pressure_mean": 0.0,
        }

    target_rate = (df["y"] == "yes").mean()
    return {
        "records": int(df.shape[0]),
        "features": len(FEATURE_COLUMNS),
        "subscription_rate": _format_percent(target_rate),
        "avg_age": round(df["age"].mean(), 1),
        "median_balance": round(df["balance"].median(), 0),
        "campaign_pressure_mean": round(df["campaign"].mean(), 2),
    }


def _segment_insights(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    enriched = df.copy()
    enriched["target"] = (enriched["y"] == "yes").astype(int)

    def top_segment(column: str) -> dict:
        grouped = (
            enriched.groupby(column)["target"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values(["mean", "count"], ascending=[False, False])
        )
        best = grouped.iloc[0]
        return {
            "dimension": column,
            "segment": str(best[column]),
            "conversion_rate": _format_percent(float(best["mean"])),
            "sample_size": int(best["count"]),
        }

    return [
        top_segment("job"),
        top_segment("education"),
        top_segment("contact"),
    ]


def _model_leaderboard(metrics_df: pd.DataFrame) -> list[dict]:
    if metrics_df.empty:
        return []

    records = []
    best_f1 = metrics_df["f1"].max()
    for row in metrics_df.sort_values("f1", ascending=False).to_dict(orient="records"):
        records.append(
            {
                "model": row["model"],
                "accuracy": _format_percent(float(row["accuracy"])),
                "f1": _format_percent(float(row["f1"])),
                "roc_auc": _format_percent(float(row["roc_auc"])),
                "is_best": float(row["f1"]) == float(best_f1),
            }
        )
    return records


def _preparation_overview(summary: dict, analysis_df: pd.DataFrame) -> dict:
    if not summary:
        return {
            "status": "Unavailable",
            "numeric_transformations": [],
            "categorical_transformations": [],
            "excluded_features": MODELING_EXCLUDED_FEATURES,
            "feature_decisions": [],
        }

    feature_decisions = []
    if not analysis_df.empty:
        columns = ["feature", "decision", "leakage_risk", "preprocessing", "rationale"]
        feature_decisions = analysis_df[columns].to_dict(orient="records")

    return {
        "status": "Ready",
        "raw_rows": summary.get("raw_rows", 0),
        "clean_rows": summary.get("clean_rows", 0),
        "dropped_duplicates": summary.get("dropped_duplicates", 0),
        "numeric_transformations": summary.get("numeric_transformations", []),
        "categorical_transformations": summary.get("categorical_transformations", []),
        "normalization_note": summary.get("normalization_note", ""),
        "excluded_features": summary.get("excluded_features", MODELING_EXCLUDED_FEATURES),
        "feature_decisions": feature_decisions,
    }


def _scoring_result(frame: pd.DataFrame) -> tuple[list[int], list[float]]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline not available.")
    predictions = model.predict(frame[FEATURE_COLUMNS]).astype(int).tolist()
    probabilities = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1].tolist()
    return predictions, probabilities


model = load_model()
metrics_df = load_metrics()
dataset_df = load_dataset()
feature_analysis_df = load_feature_analysis()
preparation_summary = load_preparation_summary()


class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str


class ScenarioComparisonRequest(BaseModel):
    baseline: CustomerData
    scenario: CustomerData


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "mlflow": describe_mlflow_runtime(),
    }


@app.get("/dashboard/overview")
def dashboard_overview() -> dict:
    dataset_summary = _dataset_overview(dataset_df)
    leaderboard = _model_leaderboard(metrics_df)
    best_model = leaderboard[0] if leaderboard else None
    return {
        "project": {
            "title": "Bank Marketing Subscription Intelligence Platform",
            "subtitle": "An end-to-end MLOps case study combining predictive modeling, MLflow experiment tracking, API deployment, and executive BI storytelling.",
        },
        "dataset": dataset_summary,
        "model_governance": {
            "tracked_models": len(leaderboard),
            "best_model": best_model["model"] if best_model else "Unavailable",
            "best_f1": best_model["f1"] if best_model else 0.0,
            "best_roc_auc": best_model["roc_auc"] if best_model else 0.0,
            "api_status": "Operational" if model is not None else "Degraded",
            "feature_set": "Pre-contact scoring",
            "excluded_features": MODELING_EXCLUDED_FEATURES,
            "leakage_note": "The production model excludes post-call duration to avoid target leakage and preserve realistic campaign planning performance.",
        },
        "data_preparation": _preparation_overview(preparation_summary, feature_analysis_df),
        "segment_insights": _segment_insights(dataset_df),
        "model_leaderboard": leaderboard,
        "feature_schema": {
            "columns": FEATURE_COLUMNS,
            "numeric_fields": NUMERIC_FIELDS,
            "categorical_options": FEATURE_OPTIONS,
            "default_form": DEFAULT_FORM,
        },
        "links": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "health": "/health",
        },
    }


@app.get("/dashboard/explainability")
def dashboard_explainability() -> dict:
    top_features = _get_feature_importance_records()
    baseline_profile = DEFAULT_FORM.copy()
    return {
        "model_name": _classifier_name(),
        "top_features": top_features,
        "strategic_recommendations": _strategic_recommendations(baseline_profile),
        "narrative": {
            "headline": "Model explainability for business stakeholders",
            "summary": "The dashboard highlights which engineered variables drive subscription propensity and which operational levers can most improve conversion likelihood.",
        },
    }


@app.get("/dashboard/data-preparation")
def dashboard_data_preparation() -> dict:
    return _preparation_overview(preparation_summary, feature_analysis_df)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")


@app.post("/predict")
def predict(data: CustomerData) -> dict:
    input_df = pd.DataFrame([data.model_dump()])
    predictions, probabilities = _scoring_result(input_df)
    pred = predictions[0]
    probability_yes = float(probabilities[0])
    return {
        "prediction": pred,
        "probability_yes": probability_yes,
        "probability_no": 1 - probability_yes,
        "label": "yes" if pred == 1 else "no",
        "risk_band": _risk_band(probability_yes),
        "recommended_action": _recommended_action(probability_yes),
    }


@app.post("/predict/what-if")
def compare_scenarios(payload: ScenarioComparisonRequest) -> dict:
    baseline_dict = payload.baseline.model_dump()
    scenario_dict = payload.scenario.model_dump()

    baseline_frame = pd.DataFrame([baseline_dict])
    scenario_frame = pd.DataFrame([scenario_dict])

    baseline_predictions, baseline_probabilities = _scoring_result(baseline_frame)
    scenario_predictions, scenario_probabilities = _scoring_result(scenario_frame)

    baseline_probability = float(baseline_probabilities[0])
    scenario_probability = float(scenario_probabilities[0])
    uplift = scenario_probability - baseline_probability

    return {
        "baseline": {
            "label": "yes" if baseline_predictions[0] == 1 else "no",
            "probability_yes": baseline_probability,
            "risk_band": _risk_band(baseline_probability),
        },
        "scenario": {
            "label": "yes" if scenario_predictions[0] == 1 else "no",
            "probability_yes": scenario_probability,
            "risk_band": _risk_band(scenario_probability),
        },
        "delta": {
            "probability_uplift": uplift,
            "probability_uplift_percent_points": round(uplift * 100, 2),
            "recommendation": _recommended_action(scenario_probability),
            "changed_fields": _scenario_delta_summary(baseline_dict, scenario_dict),
        },
    }


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)) -> dict:
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content), sep=None, engine="python")
        df.columns = [str(column).strip().strip('"') for column in df.columns]

        # Accept legacy batch files that still contain post-call fields such as duration.
        for optional_column in ["duration", "y"]:
            if optional_column in df.columns:
                df = df.drop(columns=[optional_column])

        _validate_columns(df)
        predictions, probabilities = _scoring_result(df.copy())
        scored = df.copy()
        scored["prediction"] = predictions
        scored["probability_yes"] = probabilities
        scored["label"] = ["yes" if value == 1 else "no" for value in predictions]
        scored["risk_band"] = [_risk_band(value) for value in probabilities]

        positive_probabilities = [value for value in probabilities if value >= 0.5]
        return {
            "summary": {
                "rows_scored": len(scored),
                "predicted_yes": int(sum(predictions)),
                "predicted_no": int(len(predictions) - sum(predictions)),
                "yes_rate_percent": round(mean(predictions) * 100, 2) if predictions else 0.0,
                "avg_probability_yes": round(mean(probabilities), 4) if probabilities else 0.0,
                "max_probability_yes": round(max(probabilities), 4) if probabilities else 0.0,
                "avg_positive_probability": round(mean(positive_probabilities), 4)
                if positive_probabilities
                else 0.0,
            },
            "records": scored.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
