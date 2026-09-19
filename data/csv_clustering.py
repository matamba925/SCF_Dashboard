"""Run the dashboard's clustering workflow on any CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_numeric_data(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV and return its numeric columns with missing values filled."""
    data = pd.read_csv(csv_path)
    numeric_data = data.select_dtypes(include="number").copy()
    numeric_data = numeric_data.replace([float("inf"), float("-inf")], pd.NA)
    numeric_data = numeric_data.dropna(axis="columns", how="all")
    numeric_data = numeric_data.fillna(numeric_data.median(numeric_only=True))

    if numeric_data.shape[1] < 2:
        raise ValueError("The CSV must contain at least two usable numeric columns.")
    if numeric_data.empty:
        raise ValueError("The CSV contains no usable rows.")

    return numeric_data


def get_high_var_features(data: pd.DataFrame, feature_count: int = 5) -> list[str]:
    """Return up to five numeric columns with the highest variance."""
    variances = data.var(axis=0, ddof=0).dropna()
    features = variances.sort_values(ascending=False).head(feature_count).index.tolist()
    if len(features) < 2:
        raise ValueError("At least two non-constant numeric features are required.")
    return features


def get_pca_labels(data: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Cluster the five highest-variance features and return a 2D PCA result."""
    features = get_high_var_features(data)
    feature_values = data[features]

    if k < 2 or k >= len(feature_values):
        raise ValueError("k must be between 2 and one less than the number of rows.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10)),
        ]
    )
    scaled_features = model.named_steps["scaler"].fit_transform(feature_values)
    model.named_steps["kmeans"].fit(scaled_features)
    components = PCA(n_components=2, random_state=42).fit_transform(scaled_features)

    return pd.DataFrame(
        {
            "PC1": components[:, 0],
            "PC2": components[:, 1],
            "labels": model.named_steps["kmeans"].labels_,
        },
        index=data.index,
    )


def cluster_csv(csv_path: str | Path, k: int = 3) -> pd.DataFrame:
    """Load a CSV and return its PCA coordinates and K-means labels."""
    return get_pca_labels(load_numeric_data(csv_path), k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to the input CSV file")
    parser.add_argument("-k", type=int, default=3, help="Number of K-means clusters")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("pca_clusters.csv"),
        help="Path for the output CSV",
    )
    args = parser.parse_args()

    result = cluster_csv(args.csv_path, k=args.k)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
