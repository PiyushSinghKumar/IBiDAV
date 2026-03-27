#!/usr/bin/env python3
"""
Data quality validation and curation script.

This script validates multi_labels quality, identifies missing labels,
and provides recommendations for improvement.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "Streamlit_app" / "ensemble_results.csv"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the ensemble results CSV."""
    return pd.read_csv(path)


def analyze_label_coverage(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze multi_labels completeness and distribution."""
    if "multi_labels" not in df.columns:
        return {"error": "multi_labels column not found"}

    total_rows = len(df)
    missing_labels = df["multi_labels"].isna().sum()
    non_empty_labels = df[df["multi_labels"].notna()].shape[0]
    coverage = non_empty_labels / total_rows if total_rows > 0 else 0.0

    # Analyze label diversity
    label_counts = {}
    for labels_str in df["multi_labels"].dropna():
        if isinstance(labels_str, str):
            # Assume comma or semicolon separated
            labels = [l.strip() for l in labels_str.replace(";", ",").split(",")]
            for label in labels:
                if label:
                    label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "total_rows": total_rows,
        "missing_labels": missing_labels,
        "coverage_pct": round(coverage * 100, 2),
        "unique_labels": len(label_counts),
        "label_distribution": dict(sorted(label_counts.items(), key=lambda x: -x[1])[:20]),
        "most_common_label": max(label_counts, key=label_counts.get) if label_counts else None,
        "least_common_label": min(label_counts, key=label_counts.get) if label_counts else None,
    }


def analyze_text_fields(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze content field completeness and text statistics."""
    results = {}

    for field in ["Title", "Abstract", "PMID", "PMCID"]:
        if field in df.columns:
            non_null = df[field].notna().sum()
            coverage = non_null / len(df) if len(df) > 0 else 0.0
            avg_len = (
                df[df[field].notna()][field].astype(str).str.len().mean()
                if non_null > 0
                else 0
            )
            results[field] = {
                "coverage_pct": round(coverage * 100, 2),
                "avg_length": round(avg_len, 2),
                "null_count": len(df) - non_null,
            }

    return results


def generate_quality_report(df: pd.DataFrame) -> None:
    """Generate and print comprehensive quality report."""
    print("\n" + "=" * 70)
    print("IBiDAV DATASET QUALITY REPORT")
    print("=" * 70)

    print("\n[LABEL COVERAGE ANALYSIS]")
    label_stats = analyze_label_coverage(df)
    if "error" not in label_stats:
        for key, value in label_stats.items():
            if key not in ["label_distribution"]:
                print(f"  {key:.<40} {value}")
        print("\n  Top 5 Labels:")
        dist = label_stats["label_distribution"]
        for label, count in list(dist.items())[:5]:
            pct = round(100 * count / label_stats["total_rows"], 2)
            print(f"    {label:.<35} {count:>5} ({pct}%)")
    else:
        print(f"  ERROR: {label_stats['error']}")

    print("\n[TEXT FIELD COVERAGE]")
    text_stats = analyze_text_fields(df)
    for field, stats in text_stats.items():
        print(f"\n  {field.upper()}:")
        for key, value in stats.items():
            print(f"    {key:.<35} {value}")

    print("\n[RECOMMENDATIONS FOR IMPROVEMENT]")
    label_stats = analyze_label_coverage(df)
    if "coverage_pct" in label_stats:
        coverage = label_stats["coverage_pct"]
        if coverage < 50:
            print(
                f"  ⚠️  Only {coverage}% labels coverage. Consider:"
                "\n     • Auto-label using title/abstract similarity"
                "\n     • Manual labeling effort on high-impact articles"
            )
        elif coverage < 80:
            print(
                "  ⚠️  Moderate label sparsity. Recommend:"
                "\n     • Focus labeling on high-cited articles"
                "\n     • Semi-supervised approaches for unlabeled data"
            )
        else:
            print(f"  ✓ Good label coverage at {coverage}%")

    print("\n[NEXT STEPS]")
    print("  1. Use scripts/prepare_training_data.py to create train/val/test splits")
    print("  2. Run scripts/evaluate_search_quality.py for baseline metrics")
    print("  3. Consider scripts/discover_topics.py for embedding-based analysis")

    print("\n" + "=" * 70 + "\n")


def main() -> int:
    """Run data quality validation."""
    try:
        df = load_dataset()
        print(f"Loaded {len(df)} rows from {DATASET_PATH}")
        generate_quality_report(df)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
