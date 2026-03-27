"""Semantic re-ranking module for search results."""

from __future__ import annotations

from typing import Any
import numpy as np


class SemanticRanker:
    """Re-rank search results using semantic similarity."""

    def __init__(self, nlp_processor: Any) -> None:
        """Initialize with NLP processor that has embedding support."""
        self.nlp_processor = nlp_processor
        self._embedding_cache: dict[str, list[float]] = {}

    def _get_embedding(self, text: str) -> list[float] | None:
        """Get cached or compute embedding."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        embedding = self.nlp_processor.get_embedding(text)
        if embedding:
            self._embedding_cache[text] = embedding
        return embedding

    def cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """Compute cosine similarity between two vectors."""
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)

        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def rerank_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        weight: float = 0.3,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Re-rank search results by semantic similarity.

        Args:
            query: Original query
            results: List of search results
            weight: Weight of semantic score (0-1)
            top_k: Maximum results to return

        Returns:
            Re-ranked results
        """
        if not results:
            return results

        # Get query embedding
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return results[:top_k]

        # Compute semantic scores
        reranked = []
        for result in results:
            # Combine title and abstract for embedding
            text = f"{result.get('title', '')} {result.get('abstract', '')}"
            text_embedding = self._get_embedding(text)

            if text_embedding:
                semantic_score = self.cosine_similarity(
                    query_embedding,
                    text_embedding,
                )
            else:
                semantic_score = 0.0

            # Combine BM25 score with semantic score
            bm25_score = result.get("score", 0.0)
            # Normalize scores to [0, 1]
            bm25_normalized = min(1.0, bm25_score / 100.0)  # Heuristic normalization
            semantic_normalized = semantic_score  # Already [0, 1]

            combined_score = (
                (1 - weight) * bm25_normalized + weight * semantic_normalized
            )

            result_copy = result.copy()
            result_copy["bm25_score"] = round(bm25_score, 3)
            result_copy["semantic_score"] = round(semantic_score, 3)
            result_copy["score"] = round(combined_score, 3)

            reranked.append((combined_score, result_copy))

        # Sort by combined score
        reranked.sort(key=lambda x: x[0], reverse=True)
        return [result for _, result in reranked[:top_k]]

    def extract_keyphrases(self, text: str, top_k: int = 5) -> list[str]:
        """Extract key phrases from text using embeddings."""
        # Split into sentences
        sentences = [s.strip() for s in text.replace(".", ".\n").split("\n") if s.strip()]

        if not sentences:
            return []

        # Get query embedding
        text_embedding = self._get_embedding(text)
        if not text_embedding:
            return []

        # Score sentences by similarity to overall text
        scored_sentences = []
        for sentence in sentences:
            sent_embedding = self._get_embedding(sentence)
            if sent_embedding:
                similarity = self.cosine_similarity(text_embedding, sent_embedding)
                scored_sentences.append((similarity, sentence))

        # Return top sentences as keyphrases
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return [sent for _, sent in scored_sentences[:top_k]]
