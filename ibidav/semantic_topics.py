"""BERTopic-based semantic topic discovery."""

from __future__ import annotations

from typing import Any
from pathlib import Path

try:
    from bertopic import BERTopic
    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / ".cache" / "models"


class SemanticTopicDiscovery:
    """Discover topics using BERTopic with semantic embeddings."""

    def __init__(self, num_topics: int = 8, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize BERTopic model."""
        if not HAS_BERTOPIC:
            raise ImportError("BERTopic not installed. Run: pip install bertopic")

        self.num_topics = num_topics
        self.embedding_model_name = embedding_model
        self.topic_model: BERTopic | None = None
        self.topics: dict[int, Any] = {}
        self.device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"

    def _build_embedding_backend(self) -> Any:
        """Build the embedding backend, preferring CUDA when available."""
        if HAS_SENTENCE_TRANSFORMERS:
            return SentenceTransformer(self.embedding_model_name, device=self.device)
        return self.embedding_model_name

    def discover_topics(self, texts: list[str]) -> dict[str, Any]:
        """
        Discover topics from document collection.

        Args:
            texts: List of documents

        Returns:
            Dictionary with topic information
        """
        if not texts:
            return {"error": "No documents provided"}

        try:
            # Initialize BERTopic
            embedding_backend = self._build_embedding_backend()
            self.topic_model = BERTopic(
                embedding_model=embedding_backend,
                nr_topics=self.num_topics,
                min_topic_size=10,
                language="english",
                calculate_probabilities=False,
            )

            # Fit model
            topics, probs = self.topic_model.fit_transform(texts)

            # Extract topic information
            topic_info = []
            for topic_id in sorted(set(topics)):
                if topic_id == -1:  # Skip outlier topic
                    continue

                try:
                    # Get top words per topic
                    words = self.topic_model.get_topic(topic_id)
                    top_words = [word for word, weight in words[:5]]

                    # Count documents in topic
                    doc_count = sum(1 for t in topics if t == topic_id)

                    topic_info.append({
                        "id": topic_id,
                        "label": f"Topic_{topic_id}",
                        "top_words": top_words,
                        "document_count": doc_count,
                    })
                except Exception:
                    pass

            return {
                "success": True,
                "device": self.device,
                "num_topics": len(topic_info),
                "topics": topic_info,
                "model": self.topic_model,
                "topics_array": topics,
            }

        except Exception as e:
            return {"error": str(e), "success": False}

    def save(self, path: Path | None = None) -> Path:
        """Save model to disk."""
        path = path or (MODEL_DIR / "bertopic_model")
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.topic_model:
            self.topic_model.save(str(path))

        return path

    @classmethod
    def load(cls, path: Path | None = None) -> SemanticTopicDiscovery:
        """Load model from disk."""
        path = path or (MODEL_DIR / "bertopic_model")

        instance = cls()
        if path.exists():
            try:
                from bertopic import BERTopic
                instance.topic_model = BERTopic.load(str(path))
                return instance
            except Exception:
                pass

        return instance
