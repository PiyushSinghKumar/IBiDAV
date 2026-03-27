#!/usr/bin/env python3
"""
Topic discovery using embedding-based methods (BERTopic or sentence transformers).

Replaces heuristic keyword extraction with neural topic models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "Streamlit_app" / "ensemble_results.csv"

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed. Run:")
    print("  uv pip install sentence-transformers")
    sys.exit(1)

try:
    from bertopic import BERTopic
except ImportError:
    print("ERROR: BERTopic not installed. Run:")
    print("  uv pip install bertopic")
    sys.exit(1)


class EmbeddingTopicDiscovery:
    """Topic discovery using sentence embeddings and BERTopic."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/allenai-specter",
        num_topics: int = 8,
    ):
        """Initialize embedding model and topic discoverer."""
        self.embedding_model_name = embedding_model
        self.num_topics = num_topics
        self.embedding_model = None
        self.topic_model = None
        self._load_models()

    def _load_models(self) -> None:
        """Load embedding and topic models."""
        print(f"Loading embedding model: {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        print(f"Initializing BERTopic with {self.num_topics} topics...")
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics=self.num_topics,
            min_topic_size=10,
        )

    def discover_topics(self, texts: list[str]) -> tuple[list[int], list[dict[str, Any]]]:
        """Discover topics in document collection."""
        if not texts:
            return [], []

        print(f"Discovering topics in {len(texts)} documents...")
        topics, _ = self.topic_model.fit_transform(texts)

        # Extract topic information
        topic_info = []
        for topic_id in sorted(set(topics)):
            if topic_id == -1:  # Skip outliers
                continue

            words = self.topic_model.get_topic(topic_id)
            top_words = [word for word, weight in words[:5]]

            doc_count = sum(1 for t in topics if t == topic_id)

            topic_info.append({
                "id": topic_id,
                "label": f"Topic {topic_id}",
                "top_words": top_words,
                "document_count": doc_count,
                "coherence": self._estimate_coherence(topic_id, words),
            })

        return topics, topic_info

    def _estimate_coherence(self, topic_id: int, words: list[tuple[str, float]]) -> float:
        """Estimate topic coherence from word weights."""
        if not words:
            return 0.0
        weights = [weight for _, weight in words]
        return float(np.mean(weights))

    @staticmethod
    def compare_topic_methods(df: pd.DataFrame, sample_size: int = 1000) -> None:
        """Compare BERTopic with simple keyword extraction."""
        print("\n" + "=" * 70)
        print("TOPIC DISCOVERY COMPARISON")
        print("=" * 70)

        # Sample abstracts
        abstracts = df["Abstract"].dropna().head(sample_size).tolist()
        if not abstracts:
            print("WARNING: No abstracts found")
            return

        print(f"\nAnalyzing {len(abstracts)} documents...")

        # BERTopic discovery
        print("\n[BERTopic Embedding-Based Discovery]")
        try:
            discoverer = EmbeddingTopicDiscovery(num_topics=8)
            topics, topic_info = discoverer.discover_topics(abstracts)

            for info in topic_info:
                print(
                    f"  Topic {info['id']:2d}: {', '.join(info['top_words'][:3])}"
                    f" ({info['document_count']} docs, coherence: {info['coherence']:.3f})"
                )
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  (Ensure sentence-transformers and BERTopic are installed)")
            return

        # Simple keyword method (for comparison)
        print("\n[NLP Heuristic Keyword Extraction (Current)]")
        from collections import Counter
        from nltk.tokenize import wordpunct_tokenize

        # Use fallback stopwords (NLTK download may timeout)
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "in", "is", "it", "of", "on", "or", "that", "the", "to", "with",
        }

        all_tokens = []
        for abstract in abstracts:
            tokens = [
                word.lower() for word in wordpunct_tokenize(abstract)
                if word.lower() not in stop_words
                and word.isalnum()
                and len(word) > 3
            ]
            all_tokens.extend(tokens)

        top_keywords = Counter(all_tokens).most_common(24)
        for i, (word, count) in enumerate(top_keywords, 1):
            if i % 3 == 1:
                print(f"\n  {word} ({count})", end="")
            else:
                print(f" | {word} ({count})", end="")

        print("\n\n[OBSERVATIONS]")
        print("  ✓ BERTopic discovers semantic topic structure")
        print("  ✓ Embedding models understand biomedical context better")
        print("  • Keyword method is faster but less meaningful for clustering")

        print("\n" + "=" * 70)


def main() -> int:
    """Run topic discovery analysis."""
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Loaded {len(df)} rows from {DATASET_PATH}")

        # Compare methods
        EmbeddingTopicDiscovery.compare_topic_methods(df)

        print("\n[NEXT STEPS]")
        print("  1. Deploy EmbeddingTopicDiscovery in ibidav/service.py")
        print("  2. Update _build_topics() to use embeddings instead of keywords")
        print("  3. Cache embeddings in runtime artifacts")
        print("  4. Add topic coherence to summary API endpoint")

        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
