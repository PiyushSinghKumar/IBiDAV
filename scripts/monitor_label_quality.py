#!/usr/bin/env python3
"""Monitor label quality improvements over time."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "Streamlit_app" / "ensemble_results.csv"
MONITOR_DIR = ROOT_DIR / ".cache" / "label_monitoring"


def log_quality_snapshot() -> int:
    """Log current label quality metrics."""
    try:
        print("=" * 70)
        print("LABEL QUALITY MONITORING")
        print("=" * 70)

        # Load dataset
        df = pd.read_csv(DATASET_PATH)
        print(f"\nDataset size: {len(df)} rows")

        # Calculate metrics
        labeled = df["multi_labels"].notna().sum()
        total = len(df)
        coverage = 100 * labeled / total if total > 0 else 0

        unique_labels = set()
        for labels_str in df["multi_labels"].dropna():
            if isinstance(labels_str, str):
                labels = [l.strip() for l in labels_str.replace(";", ",").split(",")]
                unique_labels.update(l for l in labels if l)

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "total_articles": total,
            "labeled_articles": labeled,
            "coverage_pct": round(coverage, 2),
            "unique_labels": len(unique_labels),
            "label_values": list(sorted(unique_labels)),
        }

        print(f"\n[CURRENT METRICS]")
        print(f"  Total articles:      {total}")
        print(f"  Labeled articles:    {labeled}")
        print(f"  Coverage:            {coverage:.2f}%")
        print(f"  Unique labels:       {len(unique_labels)}")

        # Save snapshot
        MONITOR_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = MONITOR_DIR / f"snapshot_{timestamp}.json"

        with snapshot_path.open("w") as f:
            json.dump(snapshot, f, indent=2)

        print(f"\n✓ Saved snapshot to {snapshot_path}")

        # Load history and show trend
        snapshots = sorted(MONITOR_DIR.glob("snapshot_*.json"))
        if len(snapshots) > 1:
            print(f"\n[COVERAGE TREND (last {min(5, len(snapshots))} snapshots)]")
            for snap_path in snapshots[-5:]:
                with snap_path.open() as f:
                    snap = json.load(f)
                    dt = snap["timestamp"][:10]
                    cov = snap["coverage_pct"]
                    trend = "📈" if cov > coverage * 0.99 else "➡️"
                    print(f"  {dt}: {cov:6.2f}% {trend}")

        # Calculate improvement potential
        improvement_potential = 100 - coverage
        if improvement_potential > 10:
            print(f"\n[IMPROVEMENT POTENTIAL]")
            print(f"  {improvement_potential:.1f}% of articles could be labeled")
            print(f"  Recommendation: Use auto-labeling with classifier")
            print(f"  Command: uv run scripts/train_label_classifier.py")

        print("\n[MONITORING BEST PRACTICES]")
        print("  1. Run monthly: uv run scripts/monitor_label_quality.py")
        print("  2. Track in version control: .cache/label_monitoring/")
        print("  3. Use for: Reporting progress, identifying bottlenecks")
        print("  4. Integrate with: CI/CD pipeline for visibility")

        print("\n" + "=" * 70 + "\n")
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(log_quality_snapshot())
