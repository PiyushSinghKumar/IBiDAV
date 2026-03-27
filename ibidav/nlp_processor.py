"""Enhanced NLP processing with spaCy and optional semantic embeddings."""

from __future__ import annotations

import re
from typing import Any

import spacy
from spacy.language import Language

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _preferred_device() -> str:
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class NLPProcessor:
    """Text processing with spaCy, lemmatization, and optional embeddings."""

    def __init__(self, enable_embeddings: bool = False):
        """Initialize spaCy model and optional embedding model."""
        self.nlp: Language | None = None
        self.embedding_model: Any = None
        self.device = _preferred_device()
        self.spacy_uses_gpu = False
        self._load_models(enable_embeddings)

    def _load_models(self, enable_embeddings: bool) -> None:
        """Load spaCy and optionally embedding model."""
        try:
            self.spacy_uses_gpu = bool(spacy.prefer_gpu())
        except Exception:
            self.spacy_uses_gpu = False

        try:
            # Only keep components needed for preprocessing to reduce latency.
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
        except OSError:
            print("WARN: spaCy model not found. Falling back to blank English tokenizer.")
            print("  Install full model with: python -m spacy download en_core_web_sm")
            self.nlp = spacy.blank("en")

        if enable_embeddings and HAS_EMBEDDINGS:
            try:
                self.embedding_model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device=self.device,
                )
            except Exception as e:
                print(f"WARN: Could not load embedding model: {e}")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text with spaCy:
        - Lowercasing
        - Tokenization
        - Lemmatization
        - Stopword removal
        - Length filtering
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.strip().lower()
        if not self.nlp:
            # Fallback to simple tokenization
            return self._simple_preprocess(text)

        try:
            doc = self.nlp(text)
            cleaned_words = [
                token.lemma_
                for token in doc
                if (
                    not token.is_stop
                    and not token.is_punct
                    and token.is_alpha
                    and len(token.text) >= 3
                )
            ]
            return " ".join(cleaned_words)
        except Exception:
            return self._simple_preprocess(text)

    @staticmethod
    def _simple_preprocess(text: str) -> str:
        """Fallback preprocessing without spaCy."""
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        words = [w for w in text.split() if len(w) >= 3]
        return " ".join(words)

    def extract_entities(self, text: str) -> list[dict[str, str]]:
        """Extract named entities from text."""
        if not text or not self.nlp:
            return []

        try:
            doc = self.nlp(text)
            return [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
            ]
        except Exception:
            return []

    def get_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        """Generate semantic embeddings for texts."""
        if not self.embedding_model:
            return None

        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception:
            return None

    def get_embedding(self, text: str) -> list[float] | None:
        """Generate semantic embedding for single text."""
        if not self.embedding_model:
            return None

        try:
            embedding = self.embedding_model.encode([text])
            return embedding[0].tolist()
        except Exception:
            return None

    def runtime_info(self) -> dict[str, Any]:
        """Return the active NLP runtime configuration."""
        gpu_name = None
        if HAS_TORCH and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

        return {
            "device": self.device,
            "spacy_uses_gpu": self.spacy_uses_gpu,
            "embeddings_enabled": self.embedding_model is not None,
            "torch_available": HAS_TORCH,
            "cuda_available": bool(HAS_TORCH and torch.cuda.is_available()),
            "gpu_name": gpu_name,
        }


# Shared global instance
_nlp_processor: NLPProcessor | None = None


def get_nlp_processor(enable_embeddings: bool = False) -> NLPProcessor:
    """Get or initialize the global NLP processor."""
    global _nlp_processor
    if _nlp_processor is None:
        _nlp_processor = NLPProcessor(enable_embeddings=enable_embeddings)
    return _nlp_processor
