#!/usr/bin/env python3
"""Train BERTopic semantic topics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "Streamlit_app" / "ensemble_results.csv"

try:
    from ibidav.semantic_topics import SemanticTopicDiscovery
except ImportError:
    print("ERROR: BERTopic not installed. Run:")
    print("  pip install bertopic")
    sys.exit(1)


def train_semantic_topics() -> int:
    """Train BERTopic on article corpus."""
    try:
        print("=" * 70)
        print("TRAINING SEMANTIC TOPICS WITH BERTOPIC")
        print("=" * 70)

        # Load dataset
        df = pd.read_csv(DATASET_PATH)
        print(f"\nLoaded {len(df)} articles")

        # Get abstracts (sample for speed)
        abstracts = df["Abstract"].dropna().head(2000).tolist()
        print(f"Using {len(abstracts)} abstracts for training")

        if len(abstracts) < 100:
            print("ERROR: Not enough abstracts")
            return 1

        print("\nTraining BERTopic (this may take several minutes)...")
        discoverer = SemanticTopicDiscovery(num_topics=8)
        result = discoverer.discover_topics(abstracts)

        if "error" in result:
            print(f"ERROR: {result['error']}")
            return 1

        print(f"\nUsing device: {result.get('device', 'unknown')}")

        print("\n[TOPICS DISCOVERED]")
        for topic in result["topics"]:
            words = ", ".join(topic["top_words"][:3])
            print(
                f"  Topic {topic['id']:2d}: {words:.<40} "
                f"({topic['document_count']} docs)"
            )

        # Save model
        save_path = discoverer.save()
        print(f"\n✓ Saved BERTopic model to {save_path}")

        print("\n[BENEFITS OVER KEYWORD EXTRACTION]")
        print("  ✓ Discovers semantic themes, not just frequent words")
        print("  ✓ Handles synonyms and word variations")
        print("  ✓ Better for exploratory analysis")
        print("  ✓ Dynamically assigns documents to topics")

        print("\n[NEXT STEPS]")
        print("  1. Integrate semantic topics into ibidav/service.py")
        print("  2. Replace keyword-based _build_topics() method")
        print("  3. Update artifact caching for semantic topics")

        print("\n" + "=" * 70 + "\n")
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(train_semantic_topics())
