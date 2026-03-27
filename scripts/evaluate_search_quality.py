#!/usr/bin/env python3
"""
Offline evaluation framework for search quality.

Measures search ranking quality using BM25 and provides baselines
for improving retrieval performance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "Streamlit_app" / "ensemble_results.csv"


@dataclass
class RankingMetrics:
    """Ranking quality metrics."""
    mean_reciprocal_rank: float
    precision_at_k: dict[int, float]
    ndcg: float
    coverage: float


class SearchEvaluator:
    """Evaluate search ranking quality offline."""

    def __init__(self, bm25_model: BM25Okapi, articles: list[dict[str, Any]]):
        """Initialize with BM25 model and articles."""
        self.bm25 = bm25_model
        self.articles = articles
        self.article_index = {
            str(art.get("article_id")): art for art in articles
        }

    def evaluate_query(
        self,
        query: str,
        tokens: list[str],
        relevant_ids: set[str],
        k: int = 10,
    ) -> dict[str, Any]:
        """Evaluate ranking for single query."""
        if not relevant_ids:
            return {"error": "No relevant documents"}

        # Get BM25 scores
        scores = self.bm25.get_scores(tokens)
        ranked_indices = np.argsort(scores)[::-1][:k]

        # Extract ranked articles
        ranked_ids = [
            str(self.articles[idx].get("article_id"))
            for idx in ranked_indices
        ]

        # Calculate metrics
        hits = sum(1 for rid in ranked_ids if rid in relevant_ids)
        precision = hits / k if k > 0 else 0
        mrr = self._calculate_mrr(ranked_ids, relevant_ids)
        ndcg = self._calculate_ndcg(ranked_ids, relevant_ids, k)

        return {
            "query": query,
            "k": k,
            "precision@k": precision,
            "hits": hits,
            "mrr": mrr,
            "ndcg": ndcg,
            "ranked_ids": ranked_ids,
        }

    @staticmethod
    def _calculate_mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
        """Calculate mean reciprocal rank."""
        for i, rid in enumerate(ranked_ids, 1):
            if rid in relevant_ids:
                return 1.0 / i
        return 0.0

    @staticmethod
    def _calculate_ndcg(
        ranked_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        """Calculate normalized discounted cumulative gain."""
        # DCG: sum of relevance/log(position)
        dcg = sum(
            1.0 / np.log2(i + 1)
            for i, rid in enumerate(ranked_ids[:k])
            if rid in relevant_ids
        )

        # Ideal DCG: perfect ranking of min(relevant_count, k)
        ideal_k = min(len(relevant_ids), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(ideal_k))

        return dcg / idcg if idcg > 0 else 0.0

    def batch_evaluate(
        self,
        queries: list[tuple[str, list[str]]],
        relevant_mapping: dict[str, set[str]],
    ) -> list[dict[str, Any]]:
        """Evaluate multiple queries."""
        results = []
        for query, tokens in queries:
            relevant_ids = relevant_mapping.get(query, set())
            result = self.evaluate_query(query, tokens, relevant_ids)
            results.append(result)
        return results

    def aggregate_metrics(self, results: list[dict[str, Any]]) -> RankingMetrics:
        """Aggregate metrics across queries."""
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            return RankingMetrics(0.0, {}, 0.0, 0.0)

        mrrs = [r["mrr"] for r in valid_results]
        ndcgs = [r["ndcg"] for r in valid_results]
        precisions = [r["precision@k"] for r in valid_results]

        coverage = len([r for r in valid_results if r["hits"] > 0]) / len(valid_results)

        return RankingMetrics(
            mean_reciprocal_rank=float(np.mean(mrrs)),
            precision_at_k={10: float(np.mean(precisions))},
            ndcg=float(np.mean(ndcgs)),
            coverage=coverage,
        )


def generate_synthetic_queries(df: pd.DataFrame, num_queries: int = 20) -> list[tuple[str, list[str]]]:
    """Generate synthetic queries from titles and abstracts."""
    queries = []

    # Extract meaningful terms from titles
    from nltk.corpus import stopwords
    from nltk.tokenize import wordpunct_tokenize

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    # Generate keyword queries from high-cited papers
    abstracts = df["Abstract"].dropna().head(50).tolist()

    for abstract in abstracts[:num_queries]:
        tokens = [
            word.lower() for word in wordpunct_tokenize(abstract)
            if word.lower() not in stop_words
            and word.isalnum()
            and 3 < len(word) < 15
        ]

        if len(tokens) >= 2:
            # Create query from sampled tokens
            selected = list(set(tokens))[:3]
            query_str = " ".join(selected)
            queries.append((query_str, selected))

    return queries[:num_queries]


def main() -> int:
    """Run search quality evaluation."""
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Loaded {len(df)} rows")

        print("\n" + "=" * 70)
        print("SEARCH QUALITY EVALUATION")
        print("=" * 70)

        # Build simple tokenized index (use smaller sample for speed)
        from nltk.tokenize import wordpunct_tokenize

        # Use fallback stopwords (NLTK download may timeout)
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "in", "is", "it", "of", "on", "or", "that", "the", "to", "with",
        }

        # Sample first 5000 docs for speed
        sample_df = df.head(5000)
        abstracts = (sample_df["Abstract"].fillna("") + " " + sample_df["Title"].fillna("")).tolist()

        corpus_tokens = []
        articles = []

        for i, (_, row) in enumerate(sample_df.iterrows()):
            text = (
                str(row.get("Abstract", "")) + " " + str(row.get("Title", ""))
            ).lower()
            tokens = [
                word for word in wordpunct_tokenize(text)
                if word not in stop_words and word.isalnum()
            ]
            corpus_tokens.append(tokens)
            articles.append({
                "article_id": i,
                "title": row.get("Title", ""),
                "pmid": row.get("PMID", ""),
            })

        # Build BM25
        print("\nBuilding BM25 index...")
        bm25 = BM25Okapi(corpus_tokens)

        # Create evaluator
        evaluator = SearchEvaluator(bm25, articles)

        # Generate synthetic queries
        print("Generating synthetic test queries...")
        queries = generate_synthetic_queries(sample_df)

        if not queries:
            print("WARNING: Could not generate test queries")
            return 0

        print(f"Generated {len(queries)} queries")

        # Create relevance mapping (simplified: articles matching query terms)
        relevant_mapping = {}
        for query, tokens in queries:
            relevant_ids = set()
            for i, doc_tokens in enumerate(corpus_tokens):
                if any(t in doc_tokens for t in tokens):
                    relevant_ids.add(str(i))
            if relevant_ids:
                relevant_mapping[query] = relevant_ids

        print(f"Queries with relevance judgments: {len(relevant_mapping)}")

        # Evaluate
        print("\nEvaluating search quality...")
        results = evaluator.batch_evaluate(queries, relevant_mapping)

        metrics = evaluator.aggregate_metrics(results)

        print("\n[AGGREGATE METRICS]")
        print(f"  Mean Reciprocal Rank (MRR):  {metrics.mean_reciprocal_rank:.3f}")
        print(f"  Precision@10:                {metrics.precision_at_k.get(10, 0):.3f}")
        print(f"  NDCG:                        {metrics.ndcg:.3f}")
        print(f"  Coverage:                    {metrics.coverage:.1%}")

        print("\n[SAMPLE RESULTS]")
        for i, result in enumerate(results[:5], 1):
            print(
                f"  [{i}] Query: '{result['query']}' "
                f"→ P@10={result['precision@k']:.2f}, "
                f"MRR={result['mrr']:.2f}"
            )

        print("\n[RECOMMENDATIONS]")
        if metrics.ndcg < 0.5:
            print("  ⚠️  NDCG < 0.5 indicates room for ranking improvement:")
            print("     • Add metadata boosts (title > abstract)")
            print("     • Use semantic embeddings for re-ranking")
            print("     • Implement query expansion")
        else:
            print("  ✓ BM25 baseline is reasonable")

        print("\n[NEXT STEPS]")
        print("  1. Improve relevance labels (use crowd-sourcing or weak supervision)")
        print("  2. Implement semantic re-ranking with embeddings")
        print("  3. Add metadata boosts and field-specific weights")
        print("  4. Integrate into CI/CD for regression testing")

        print("\n" + "=" * 70 + "\n")
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
