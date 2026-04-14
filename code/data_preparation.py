import json
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from settings import (
    CLEAN_DATA_PATH,
    FEATURE_ANALYSIS_PATH,
    MODELING_EXCLUDED_FEATURES,
    MODELING_FEATURE_COLUMNS,
    PREPARATION_SUMMARY_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def load_raw_dataset(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=";")


def clean_bank_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    prepared = df.copy()
    prepared.columns = [column.strip() for column in prepared.columns]

    object_columns = prepared.select_dtypes(include="object").columns
    for column in object_columns:
        prepared[column] = prepared[column].astype(str).str.strip()

    rows_before = int(prepared.shape[0])
    prepared = prepared.drop_duplicates().reset_index(drop=True)
    rows_after = int(prepared.shape[0])

    selected_columns = MODELING_FEATURE_COLUMNS + [TARGET_COLUMN]
    prepared = prepared[selected_columns].copy()

    summary = {
        "raw_rows": rows_before,
        "clean_rows": rows_after,
        "dropped_duplicates": rows_before - rows_after,
        "feature_count": len(MODELING_FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "excluded_features": MODELING_EXCLUDED_FEATURES,
        "numeric_transformations": ["median_imputation", "standardization"],
        "categorical_transformations": ["most_frequent_imputation", "one_hot_encoding"],
        "normalization_note": "Global normalization is not applied to tree models. Standardization is applied to numeric features and supports the linear baseline.",
    }
    return prepared, summary


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def _cramers_v(feature: pd.Series, target: pd.Series) -> float:
    contingency = pd.crosstab(feature, target)
    if contingency.empty or min(contingency.shape) < 2:
        return 0.0

    chi2, _, _, _ = chi2_contingency(contingency)
    n_obs = contingency.to_numpy().sum()
    if n_obs == 0:
        return 0.0

    phi2 = chi2 / n_obs
    rows, cols = contingency.shape
    phi2corr = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / max(n_obs - 1, 1))
    rows_corr = rows - ((rows - 1) ** 2) / max(n_obs - 1, 1)
    cols_corr = cols - ((cols - 1) ** 2) / max(n_obs - 1, 1)
    denominator = min((cols_corr - 1), (rows_corr - 1))
    if denominator <= 0:
        return 0.0
    return float((phi2corr / denominator) ** 0.5)


def generate_feature_analysis(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> pd.DataFrame:
    target = (raw_df[TARGET_COLUMN] == "yes").astype(int)
    numeric_features = [column for column in cleaned_df.select_dtypes(exclude=["object"]).columns if column != TARGET_COLUMN]
    categorical_features = [column for column in cleaned_df.select_dtypes(include=["object"]).columns if column != TARGET_COLUMN]

    rows = []
    for column in raw_df.columns:
        if column == TARGET_COLUMN:
            continue

        is_included = column in MODELING_FEATURE_COLUMNS
        feature_type = "categorical" if raw_df[column].dtype == "object" else "numeric"
        missing_rate = round(float(raw_df[column].isna().mean()) * 100, 4)
        unique_count = int(raw_df[column].nunique(dropna=True))

        if column == "duration":
            decision = "remove"
            rationale = "Post-call information causes target leakage for pre-contact scoring."
            leakage_risk = "high"
        elif column in {"day", "month", "campaign"}:
            decision = "keep_with_monitoring"
            rationale = "Operational timing variable kept for campaign planning but must match the prediction moment."
            leakage_risk = "medium"
        else:
            decision = "keep" if is_included else "remove"
            rationale = "Retained for pre-contact modeling." if is_included else "Excluded from production feature set."
            leakage_risk = "low"

        target_association = 0.0
        if column in numeric_features:
            target_association = abs(float(cleaned_df[column].corr((cleaned_df[TARGET_COLUMN] == "yes").astype(int))))
        elif column in categorical_features:
            target_association = _cramers_v(cleaned_df[column], cleaned_df[TARGET_COLUMN])

        preprocessing = "not_used"
        if is_included:
            preprocessing = "standardize" if feature_type == "numeric" else "one_hot_encode"

        rows.append(
            {
                "feature": column,
                "type": feature_type,
                "included_in_model": is_included,
                "decision": decision,
                "leakage_risk": leakage_risk,
                "missing_rate_percent": missing_rate,
                "unique_values": unique_count,
                "target_association": round(target_association, 4),
                "preprocessing": preprocessing,
                "rationale": rationale,
            }
        )

    report = pd.DataFrame(rows).sort_values(
        by=["included_in_model", "leakage_risk", "target_association"],
        ascending=[False, True, False],
    )
    return report


def prepare_datasets() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw_df = load_raw_dataset()
    cleaned_df, summary = clean_bank_dataset(raw_df)
    analysis_df = generate_feature_analysis(raw_df, cleaned_df)

    cleaned_df.to_csv(CLEAN_DATA_PATH, index=False)
    analysis_df.to_csv(FEATURE_ANALYSIS_PATH, index=False)
    PREPARATION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return raw_df, cleaned_df, summary


if __name__ == "__main__":
    _, cleaned, preparation_summary = prepare_datasets()
    print(f"Prepared dataset saved to: {CLEAN_DATA_PATH}")
    print(f"Feature analysis saved to: {FEATURE_ANALYSIS_PATH}")
    print(f"Preparation summary saved to: {PREPARATION_SUMMARY_PATH}")
    print(
        "Prepared shape:",
        cleaned.shape,
        "| Excluded features:",
        preparation_summary["excluded_features"],
    )
