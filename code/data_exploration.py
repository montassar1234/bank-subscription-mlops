import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from settings import DATA_DIR, RAW_DATA_PATH


def load_data(file_path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the bank marketing dataset."""
    return pd.read_csv(file_path)


def explore_dataset(df: pd.DataFrame) -> None:
    """Perform exploratory data analysis on the dataset."""
    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    print(f"\nTarget distribution:\n{df['y'].value_counts()}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for numeric columns."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    correlation_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap - Bank Marketing Data")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "correlation_heatmap_bank.png", dpi=140)
    plt.close()
    print("Correlation heatmap saved.")


def plot_numerical_distributions(df: pd.DataFrame) -> None:
    """Plot box plots for numeric columns."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        axes[idx].boxplot(df[col])
        axes[idx].set_title(f"Box Plot: {col}")
        axes[idx].set_ylabel(col)

    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(DATA_DIR / "numerical_boxplots_bank.png", dpi=140)
    plt.close()
    print("Numerical distributions plot saved.")


def plot_categorical_distributions(df: pd.DataFrame) -> None:
    """Plot categorical feature distributions."""
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "y" in categorical_cols:
        categorical_cols.remove("y")

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, col in enumerate(categorical_cols):
        value_counts = df[col].value_counts()
        axes[idx].barh(value_counts.index, value_counts.values)
        axes[idx].set_title(f"Distribution: {col}")
        axes[idx].set_xlabel("Count")

    for idx in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(DATA_DIR / "categorical_impact_bank.png", dpi=140)
    plt.close()
    print("Categorical distributions plot saved.")


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Plot target variable distribution."""
    target_counts = df["y"].value_counts()

    plt.figure(figsize=(8, 6))
    plt.bar(target_counts.index, target_counts.values, color=["#1f77b4", "#ff7f0e"])
    plt.title("Target Variable Distribution (y)")
    plt.xlabel("Subscription")
    plt.ylabel("Count")
    plt.xticks(["no", "yes"])

    for i, value in enumerate(target_counts.values):
        plt.text(i, value + 100, str(value), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(DATA_DIR / "target_imbalance_bank.png", dpi=140)
    plt.close()
    print("Target distribution plot saved.")


def main() -> None:
    """Main data exploration function."""
    print("Loading dataset...")
    df = load_data()

    explore_dataset(df)

    print("\nGenerating visualizations...")
    plot_correlation_heatmap(df)
    plot_numerical_distributions(df)
    plot_categorical_distributions(df)
    plot_target_distribution(df)

    print("\n" + "=" * 80)
    print("Data exploration complete! Visualizations saved to:", DATA_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
