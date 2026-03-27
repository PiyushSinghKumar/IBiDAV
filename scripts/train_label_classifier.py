#!/usr/bin/env python3
"""Train label classifier on labeled articles."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "splits"

from ibidav.label_classifier import LabelClassifier


def train_label_classifier() -> int:
    """Train label classifier on training data."""
    try:
        # Load training data
        train_path = DATA_DIR / "train.csv"
        if not train_path.exists():
            print(f"ERROR: {train_path} not found")
            print("Run: uv run scripts/prepare_training_data.py")
            return 1

        print("=" * 70)
        print("TRAINING LABEL CLASSIFIER")
        print("=" * 70)

        train_df = pd.read_csv(train_path)
        print(f"\nLoaded {len(train_df)} training samples")

        # Filter to labeled samples only
        labeled_df = train_df[train_df["multi_labels"].notna()].copy()
        print(f"Using {len(labeled_df)} labeled samples")

        if len(labeled_df) < 10:
            print("ERROR: Not enough labeled samples to train")
            return 1

        # Prepare training data
        texts = (
            labeled_df["Title"].fillna("").astype(str) +
            " " +
            labeled_df["Abstract"].fillna("").astype(str)
        ).tolist()

        labels = labeled_df["multi_labels"].tolist()

        print(f"\nTraining on {len(texts)} texts...")

        # Train classifier
        classifier = LabelClassifier()
        metrics = classifier.train(texts, labels)

        print("\n[TRAINING RESULTS]")
        for key, value in metrics.items():
            if key != "label_classes":
                print(f"  {key:.<40} {value}")

        print(f"\n[LABEL CLASSES]")
        if "label_classes" in metrics:
            for label in metrics["label_classes"]:
                print(f"  • {label}")

        # Save classifier
        save_path = classifier.save()
        print(f"\n✓ Saved classifier to {save_path}")

        # Test on sample
        print("\n[SAMPLE PREDICTIONS]")
        sample_idx = min(3, len(labeled_df) - 1)
        for i in range(sample_idx):
            text = texts[i]
            true_labels = labels[i]
            pred_labels = classifier.predict(text)

            print(f"\n  Sample {i + 1}:")
            print(f"    Text: {text[:80]}...")
            print(f"    True: {true_labels}")
            print(f"    Pred: {pred_labels}")

        print("\n[NEXT STEPS]")
        print("  1. Integrate classifier into ibidav/service.py")
        print("  2. Use for auto-labeling unlabeled articles")
        print("  3. Evaluate on test set: data/splits/test.csv")

        print("\n" + "=" * 70 + "\n")
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(train_label_classifier())
