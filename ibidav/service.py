from __future__ import annotations

import base64
import io
import pickle
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from ibidav.nlp_processor import get_nlp_processor
from ibidav.label_classifier import LabelClassifier
from ibidav.semantic_ranker import SemanticRanker
from PIL import Image, ImageDraw, ImageFont
from rank_bm25 import BM25Okapi


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "Streamlit_app" / "ensemble_results.csv"
ARTIFACT_DIR = ROOT_DIR / ".cache" / "ibidav"
ARTIFACT_PATH = ARTIFACT_DIR / "runtime_artifacts.pkl"

DEFAULT_PAGE_SIZE = 6
CATEGORY_PAGE_SIZE = 6
LOAD_MORE_PAGE_SIZE = 3
TOPIC_DOCUMENT_LIMIT = 5000
TOPIC_COUNT = 8
TOP_WORDS_PER_TOPIC = 8
ARTIFACT_VERSION = 2
ARTIFACT_VERSION = 3
USE_SEMANTIC_RANKING = True
USE_LABEL_CLASSIFIER = True


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()



def _split_query(query: str) -> list[str]:
    return [part.strip().lower() for part in re.split(r"[+\s]+", query) if part.strip()]

def _split_query(query: str) -> list[str]:
    return [part.strip().lower() for part in re.split(r"[+\s]+", query) if part.strip()]


@dataclass
class TopicSummary:
    index: int
    label: str
    words: list[str]


class IBiDAVService:
    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
        artifact_path: Path = ARTIFACT_PATH,
    ) -> None:
        self.dataset_path = dataset_path
        self.artifact_path = artifact_path
        self.nlp_processor = get_nlp_processor(enable_embeddings=USE_SEMANTIC_RANKING)
        self.semantic_ranker = SemanticRanker(self.nlp_processor) if USE_SEMANTIC_RANKING else None
        self.label_classifier = LabelClassifier.load() if USE_LABEL_CLASSIFIER else None

        self._raw_df: pd.DataFrame | None = None
        self._articles_df: pd.DataFrame | None = None
        self._article_records: list[dict[str, Any]] = []
        self._article_tokens: list[list[str]] = []
        self._article_category_map: dict[int, set[str]] = {}
        self._category_records: dict[str, list[dict[str, Any]]] = {}
        self._topics: list[TopicSummary] = []
        self._topic_word_frequencies: dict[str, int] = {}
        self._category_counts: dict[str, int] = {}
        self._stats: dict[str, Any] = {}
        self._bm25: BM25Okapi | None = None
        self._init_lock = threading.Lock()
        self._initializing = False
        self._initialization_error: str | None = None
        self._bm25_lock = threading.Lock()
        self._bm25_building = False

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._raw_df is None:
            self._initialize()
        assert self._raw_df is not None
        return self._raw_df

    @property
    def articles_df(self) -> pd.DataFrame:
        if self._articles_df is None:
            self._initialize()
        assert self._articles_df is not None
        return self._articles_df

    def _initialize(self) -> None:
        if self._load_artifact():
            return

        bundle = self.build_runtime_bundle()
        self._apply_bundle(bundle)

    def is_ready(self) -> bool:
        return self._articles_df is not None

    def warmup_async(self) -> None:
        with self._init_lock:
            if self.is_ready() or self._initializing:
                return
            self._initializing = True
            self._initialization_error = None
        thread = threading.Thread(target=self._warmup_worker, daemon=True, name="ibidav-warmup")
        thread.start()

    def _warmup_worker(self) -> None:
        try:
            self._initialize()
        except Exception as exc:
            self._initialization_error = str(exc)
        finally:
            with self._init_lock:
                self._initializing = False

    def initialization_state(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready(),
            "initializing": self._initializing,
            "error": self._initialization_error,
            "artifacts_present": self.artifact_path.exists(),
        }

    def _try_ready_from_artifact_sync(self) -> None:
        """Fast-path readiness when an artifact already exists.

        This prevents prolonged warm-up states if a background thread stalls.
        """
        if self.is_ready() or not self.artifact_path.exists():
            return
        with self._init_lock:
            if self.is_ready():
                return
            try:
                if self._load_artifact():
                    self._initialization_error = None
            except Exception as exc:
                self._initialization_error = str(exc)
            finally:
                self._initializing = False

    def _load_artifact(self) -> bool:
        if not self.artifact_path.exists():
            return False
        try:
            with self.artifact_path.open("rb") as handle:
                bundle = pickle.load(handle)
        except Exception:
            return False

        source_mtime_ns = int(self.dataset_path.stat().st_mtime_ns)
        metadata = bundle.get("metadata", {})
        if metadata.get("artifact_version") != ARTIFACT_VERSION:
            return False
        if metadata.get("source_mtime_ns") != source_mtime_ns:
            return False

        self._apply_bundle(bundle)
        return True

    def _apply_bundle(self, bundle: dict[str, Any]) -> None:
        self._raw_df = bundle["raw_df"]
        self._articles_df = bundle["articles_df"]
        self._article_records = bundle["article_records"]
        self._article_tokens = bundle["article_tokens"]
        self._article_category_map = bundle["article_category_map"]
        self._category_records = bundle["category_records"]
        self._topics = [TopicSummary(**topic) for topic in bundle["topics"]]
        self._topic_word_frequencies = bundle["topic_word_frequencies"]
        self._category_counts = bundle["category_counts"]
        self._stats = bundle["stats"]
        # Build BM25 lazily so the app becomes interactive sooner.
        self._bm25 = None

    def _ensure_bm25_async(self) -> None:
        if self._bm25 is not None or not self._article_tokens:
            return
        with self._bm25_lock:
            if self._bm25 is not None or self._bm25_building:
                return
            self._bm25_building = True
        thread = threading.Thread(target=self._build_bm25_worker, daemon=True, name="ibidav-bm25")
        thread.start()

    def _build_bm25_worker(self) -> None:
        started = time.time()
        try:
            bm25 = BM25Okapi(self._article_tokens) if self._article_tokens else None
            self._bm25 = bm25
        except Exception:
            self._bm25 = None
        finally:
            with self._bm25_lock:
                self._bm25_building = False
            elapsed = round(time.time() - started, 2)
            print(f"INFO: BM25 background build finished in {elapsed}s")

    def build_runtime_bundle(self) -> dict[str, Any]:
        df = pd.read_csv(self.dataset_path)
        df = self._normalize_dataframe(df)
        articles_df = self._build_article_dataframe(df)
        article_records = articles_df.to_dict("records")
        article_tokens = [
            record["processed_corpus"].split()
            for record in article_records
            if record["processed_corpus"]
        ]

        # Keep BM25 aligned with article order, including empty-token documents.
        article_tokens = [
            record["processed_corpus"].split() if record["processed_corpus"] else ["__empty__"]
            for record in article_records
        ]

        article_category_map = {
            int(record["article_id"]): set(record["categories"])
            for record in article_records
        }
        category_records = self._build_category_records(df)
        topics = self._build_topics(articles_df)
        topic_word_frequencies = self._build_topic_word_frequencies(articles_df["processed_corpus"], topics)
        category_counts = {category: len(records) for category, records in category_records.items()}
        labeled_articles = int(articles_df["has_multi_label"].sum())
        total_articles = int(len(articles_df))
        stats = {
            "raw_rows": int(len(df)),
            "unique_articles": total_articles,
            "labeled_articles": labeled_articles,
            "multi_label_coverage": round(labeled_articles / total_articles, 4) if total_articles else 0.0,
            "artifact_path": str(self.artifact_path),
        }

        return {
            "metadata": {
                "artifact_version": ARTIFACT_VERSION,
                "source_path": str(self.dataset_path),
                "source_mtime_ns": int(self.dataset_path.stat().st_mtime_ns),
            },
            "raw_df": df,
            "articles_df": articles_df,
            "article_records": article_records,
            "article_tokens": article_tokens,
            "article_category_map": article_category_map,
            "category_records": category_records,
            "topics": [{"index": topic.index, "label": topic.label, "words": topic.words} for topic in topics],
            "topic_word_frequencies": topic_word_frequencies,
            "category_counts": category_counts,
            "stats": stats,
        }

    def save_runtime_bundle(self, bundle: dict[str, Any] | None = None) -> Path:
        bundle = bundle or self.build_runtime_bundle()
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with self.artifact_path.open("wb") as handle:
            pickle.dump(bundle, handle)
        return self.artifact_path

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for column in [
            "PMCID",
            "PMID",
            "Title",
            "Abstract",
            "Article URL",
            "Image URL",
            "Category",
            "multi_labels",
        ]:
            if column not in df.columns:
                df[column] = ""
            df[column] = df[column].apply(_safe_text)

        df["Corpus"] = df["Title"] + " " + df["Abstract"] + " " + df["Category"]
        df["Processed_Corpus"] = df["Corpus"].map(self.preprocess_text)
        df["Search_Text"] = (
            df["PMID"] + " " + df["PMCID"] + " " + df["Title"] + " " + df["Abstract"] + " " + df["Category"]
        ).str.lower()
        return df

    def _build_article_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        article_keys = ["PMCID", "PMID", "Title", "Abstract", "Article URL"]
        article_df = (
            df.groupby(article_keys, dropna=False)
            .agg(
                {
                    "Category": lambda values: sorted({value for value in values if value}),
                    "multi_labels": lambda values: sorted({value for value in values if value}),
                    "Image URL": lambda values: sorted({value for value in values if value}),
                    "Processed_Corpus": "first",
                    "Search_Text": "first",
                }
            )
            .reset_index()
        )
        article_df["article_id"] = range(len(article_df))
        article_df["primary_category"] = article_df["Category"].map(lambda values: values[0] if values else "")
        article_df["has_multi_label"] = article_df["multi_labels"].map(bool)
        article_df = article_df.rename(
            columns={
                "PMCID": "pmcid",
                "PMID": "pmid",
                "Title": "title",
                "Abstract": "abstract",
                "Article URL": "article_url",
                "Category": "categories",
                "Image URL": "image_urls",
                "Processed_Corpus": "processed_corpus",
                "Search_Text": "search_text",
            }
        )
        return article_df

    def _build_category_records(self, df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        category_records: dict[str, list[dict[str, Any]]] = {}
        for category in sorted(value for value in df["Category"].unique() if value):
            grouped_df = (
                df[df["Category"] == category]
                .groupby(["PMCID", "PMID", "Title", "Abstract", "Article URL"], dropna=False)
                .agg({"Image URL": lambda values: sorted({value for value in values if value})})
                .reset_index()
            )
            records: list[dict[str, Any]] = []
            for _, row in grouped_df.iterrows():
                abstract = _safe_text(row["Abstract"])
                images = row["Image URL"] if isinstance(row["Image URL"], list) else []
                records.append(
                    {
                        "pmcid": _safe_text(row["PMCID"]),
                        "pmid": _safe_text(row["PMID"]),
                        "title": _safe_text(row["Title"]),
                        "abstract": abstract,
                        "abstract_preview": abstract[:160] + ("..." if len(abstract) > 160 else ""),
                        "article_url": _safe_text(row["Article URL"]),
                        "image_urls": images[:6],
                        "image_count": len(images),
                        "search_text": (
                            f"{_safe_text(row['PMCID'])} {_safe_text(row['PMID'])} "
                            f"{_safe_text(row['Title'])} {abstract} {category}"
                        ).lower(),
                    }
                )
            category_records[category] = records
        return category_records

    def preprocess_text(self, text: str) -> str:
        """Preprocess text using spaCy NLP processor."""
        return self.nlp_processor.preprocess_text(_safe_text(text))

    def _build_topics(self, articles_df: pd.DataFrame) -> list[TopicSummary]:
        if articles_df.empty:
            return []

        topics: list[TopicSummary] = []
        category_counts: Counter[str] = Counter()
        for categories in articles_df["categories"]:
            for category in categories:
                category_counts[category] += 1

        top_categories = [category for category, _ in category_counts.most_common(TOPIC_COUNT)]
        used_words: set[str] = set()
        for topic_index, category in enumerate(top_categories):
            category_docs = articles_df[
                articles_df["categories"].map(lambda values: category in values)
            ]["processed_corpus"].head(TOPIC_DOCUMENT_LIMIT)
            word_counts: Counter[str] = Counter()
            for doc in category_docs:
                if not isinstance(doc, str):
                    continue
                for word in doc.split():
                    if len(word) <= 3 or word in used_words:
                        continue
                    word_counts[word] += 1
            words = [word for word, _ in word_counts.most_common(TOP_WORDS_PER_TOPIC)]
            used_words.update(words)
            topics.append(TopicSummary(index=topic_index, label=category, words=words))
        return topics

    def _build_topic_word_frequencies(
        self,
        processed_corpus: pd.Series,
        topics: list[TopicSummary],
    ) -> dict[str, int]:
        topic_words = {word for topic in topics for word in topic.words}
        frequencies: Counter[str] = Counter()
        if not topic_words:
            return {}

        for text in processed_corpus:
            if not isinstance(text, str):
                continue
            for word in text.split():
                if word in topic_words:
                    frequencies[word] += 1
        return dict(frequencies)

    def topic_wordcloud_base64(self) -> str:
        self.articles_df
        frequencies = self._topic_word_frequencies
        if not frequencies:
            return ""
        image = self._build_keyword_poster(frequencies)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _build_keyword_poster(self, frequencies: dict[str, int]) -> Image.Image:
        width = 1200
        height = 600
        image = Image.new("RGB", (width, height), "#081120")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        palette = ["#f4efe7", "#f6bd60", "#84dcc6", "#a7c957", "#f28482"]
        items = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)[:24]
        if not items:
            return image

        max_freq = items[0][1]
        min_freq = items[-1][1]
        columns = 4
        card_width = width // columns
        row_height = 92

        for index, (word, freq) in enumerate(items):
            row = index // columns
            column = index % columns
            x = 28 + column * card_width
            y = 24 + row * row_height
            scale = 1.0 if max_freq == min_freq else (freq - min_freq) / (max_freq - min_freq)
            text = word.upper()
            color = palette[index % len(palette)]
            draw.text((x, y), text, fill=color, font=font)
            draw.text((x, y + 18), f"score {freq}", fill="#c7d0db", font=font)
            bar_width = int(36 + scale * (card_width - 92))
            draw.rounded_rectangle((x, y + 40, x + bar_width, y + 56), radius=7, fill=color)
        return image

    def category_counts(self) -> dict[str, int]:
        if not self.is_ready():
            self._try_ready_from_artifact_sync()
        if not self.is_ready():
            self.warmup_async()
            return {}
        return self._category_counts

    def topics(self) -> list[dict[str, Any]]:
        if not self.is_ready():
            self._try_ready_from_artifact_sync()
        if not self.is_ready():
            self.warmup_async()
            return []
        return [{"index": topic.index, "label": topic.label, "words": topic.words} for topic in self._topics]

    def stats(self) -> dict[str, Any]:
        if not self.is_ready():
            self._try_ready_from_artifact_sync()
        if not self.is_ready():
            self.warmup_async()
            return {}
        return self._stats

    def search(self, query: str, limit: int = 100) -> list[dict[str, Any]]:
        if not self.is_ready():
            self._try_ready_from_artifact_sync()
        if not self.is_ready():
            self.warmup_async()
            return []
        normalized_query = query.strip()
        if not normalized_query:
            return []

        self._ensure_bm25_async()
        query_terms = _split_query(normalized_query)
        processed_query = self.preprocess_text(normalized_query).split()
        scores = self._bm25.get_scores(processed_query) if self._bm25 and processed_query else [0.0] * len(self._article_records)

        ranked_results: list[tuple[float, dict[str, Any]]] = []
        for index, record in enumerate(self._article_records):
            exact_score = self._exact_match_score(record, query_terms)
            bm25_score = float(scores[index]) if index < len(scores) else 0.0
            final_score = exact_score + bm25_score
            if final_score <= 0:
                continue

            enriched_record = {
                "pmcid": record["pmcid"],
                "pmid": record["pmid"],
                "title": record["title"],
                "abstract": record["abstract"],
                "article_url": record["article_url"],
                "category": record["primary_category"],
                "categories": record["categories"],
                "score": round(final_score, 3),
            }
            ranked_results.append((final_score, enriched_record))

        ranked_results.sort(key=lambda item: item[0], reverse=True)
        results = [record for _, record in ranked_results[:min(limit * 2, len(ranked_results))]]

        # Apply semantic re-ranking if enabled
        if self.semantic_ranker and len(results) > 0:
            results = self.semantic_ranker.rerank_results(
                query,
                results,
                weight=0.25,
                top_k=limit,
            )

        return results[:limit]

    def _exact_match_score(self, record: dict[str, Any], query_terms: list[str]) -> float:
        """Score based on exact term matches with metadata boosts."""
        if not query_terms:
            return 0.0
        title = record["title"].lower()
        abstract = record["abstract"].lower()
        pmid = record["pmid"].lower()
        pmcid = record["pmcid"].lower()
        categories = " ".join(record["categories"]).lower()

        score = 0.0
        for term in query_terms:
            # PMID/PMCID matches are highest priority
            if term in pmid or term in pmcid:
                score += 10.0
            # Title matches are high priority (boosted from 5.0 to 7.0)
            elif term in title:
                score += 7.0
            # Category matches
            elif term in categories:
                score += 4.0
            # Abstract matches are lower priority
            elif term in abstract:
                score += 1.5
        return score

    def category_results(
        self,
        category: str,
        query: str = "",
        offset: int = 0,
        limit: int = CATEGORY_PAGE_SIZE,
    ) -> dict[str, Any]:
        if not self.is_ready():
            self._try_ready_from_artifact_sync()
        if not self.is_ready():
            self.warmup_async()
            return {
                "category": category,
                "results": [],
                "total": 0,
                "offset": offset,
                "limit": limit,
                "remaining": 0,
                "next_offset": 0,
                "warming_up": True,
            }
        records = list(self._category_records.get(category, []))
        if not records:
            return {
                "category": category,
                "results": [],
                "total": 0,
                "offset": offset,
                "limit": limit,
                "remaining": 0,
                "next_offset": 0,
            }

        normalized_query = query.strip().lower()
        if normalized_query:
            terms = _split_query(normalized_query)
            records = [
                record
                for record in records
                if any(term in record["search_text"] for term in terms)
            ]

        total = len(records)
        page_records = records[offset : offset + limit]
        results = [{key: value for key, value in record.items() if key != "search_text"} for record in page_records]
        return {
            "category": category,
            "results": results,
            "total": total,
            "offset": offset,
            "limit": limit,
            "remaining": max(0, total - (offset + len(results))),
            "next_offset": min(total, offset + len(results)),
        }

    def summary(self) -> dict[str, Any]:
        if not self.is_ready():
            self._try_ready_from_artifact_sync()
        if not self.is_ready():
            self.warmup_async()
            init_state = self.initialization_state()
            return {
                "dataset_path": str(self.dataset_path),
                "article_rows": 0,
                "unique_articles": 0,
                "category_counts": {},
                "topics": [],
                "wordcloud_base64": "",
                "default_page_size": DEFAULT_PAGE_SIZE,
                "load_more_page_size": LOAD_MORE_PAGE_SIZE,
                "category_count": 0,
                "stats": {},
                "artifacts_present": init_state["artifacts_present"],
                "warming_up": True,
                "init_state": init_state,
                "message": "Data is warming up. Please wait a moment and retry.",
            }
        return {
            "dataset_path": str(self.dataset_path),
            "article_rows": int(len(self.dataframe)),
            "unique_articles": int(len(self.articles_df)),
            "category_counts": self.category_counts(),
            "topics": self.topics(),
            "wordcloud_base64": self.topic_wordcloud_base64(),
            "default_page_size": DEFAULT_PAGE_SIZE,
            "load_more_page_size": LOAD_MORE_PAGE_SIZE,
            "category_count": len(self._category_counts),
            "stats": self.stats(),
            "artifacts_present": self.artifact_path.exists(),
        }


service = IBiDAVService()
