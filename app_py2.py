"""Apply the dashboard clustering workflow to any CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.csv_clustering import cluster_csv


def process_csv(csv_path: str | Path, k: int = 3):
    """Return PCA coordinates and K-means labels for a CSV file."""
    return cluster_csv(csv_path, k=k)


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

    result = process_csv(args.csv_path, k=args.k)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
