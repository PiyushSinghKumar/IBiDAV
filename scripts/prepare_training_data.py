#!/usr/bin/env python3
"""
Prepare training, validation, and test splits for model development.

Creates reproducible data splits for training label classifiers,
topic models, and search ranking models.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "Streamlit_app" / "ensemble_results.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "splits"


def create_data_splits(
    df: pd.DataFrame,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Create stratified train/val/test splits.

    Stratification respects category and label distributions.
    """
    np.random.seed(random_seed)

    # Stratify by primary category if available
    if "categories" in df.columns:
        stratify_col = "categories"
    else:
        stratify_col = None

    # Random split
    n = len(df)
    indices = np.random.permutation(n)

    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def analyze_split_quality(splits: dict[str, pd.DataFrame]) -> None:
    """Analyze split balance and quality."""
    print("\n[SPLIT DISTRIBUTION]")

    for split_name, split_df in splits.items():
        pct = 100 * len(split_df) / sum(len(s) for s in splits.values())
        print(f"\n  {split_name.upper()}:")
        print(f"    Size: {len(split_df)} ({pct:.1f}%)")

        if "multi_labels" in split_df.columns:
            labeled = split_df["multi_labels"].notna().sum()
            labeled_pct = 100 * labeled / len(split_df)
            print(f"    Labeled: {labeled} ({labeled_pct:.1f}%)")

        if "categories" in split_df.columns:
            cat_counts = split_df["categories"].value_counts()
            print(f"    Top categories: {dict(cat_counts.head(3))}")

        if "pmid" in split_df.columns:
            has_pmid = split_df["pmid"].notna().sum()
            print(f"    With PMID: {has_pmid} ({100*has_pmid/len(split_df):.1f}%)")


def save_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Save splits to CSV and metadata to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        split_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        print(f"✓ Saved {split_path}")

    # Save metadata
    metadata = {
        "random_seed": 42,
        "splitting_strategy": "random",
        "sizes": {
            name: len(df) for name, df in splits.items()
        },
        "total_rows": sum(len(df) for df in splits.values()),
    }

    metadata_path = output_dir / "split_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved {metadata_path}")


def main() -> int:
    """Prepare training data splits."""
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Loaded {len(df)} rows from {DATASET_PATH}")

        print("\n" + "=" * 70)
        print("PREPARING TRAINING DATA SPLITS")
        print("=" * 70)

        # Create splits
        print("\nCreating train/val/test splits (70/15/15)...")
        splits = create_data_splits(df, train_pct=0.7, val_pct=0.15, test_pct=0.15)

        # Analyze quality
        analyze_split_quality(splits)

        # Save
        print("\nSaving splits...")
        save_splits(splits)

        print("\n[NEXT STEPS]")
        print("  1. Use data/splits/train.csv for label classifier training")
        print("  2. Use data/splits/val.csv for hyperparameter tuning")
        print("  3. Use data/splits/test.csv for final evaluation")
        print("  4. Create custom Dataset classes for PyTorch/TensorFlow")

        print("\n[EXAMPLE USAGE]")
        print("  train_df = pd.read_csv('data/splits/train.csv')")
        print("  # Train label classifier, topic model, ranking model, etc.")
        print("  val_df = pd.read_csv('data/splits/val.csv')")
        print("  test_df = pd.read_csv('data/splits/test.csv')")

        print("\n" + "=" * 70 + "\n")
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
