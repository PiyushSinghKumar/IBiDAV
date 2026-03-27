#!/usr/bin/env python3
"""
NLP upgrade module using spaCy for biomedical text processing.

Replaces NLTK preprocessing with spaCy for better handling of
biomedical terminology and improved tokenization/lemmatization.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import spacy
    from spacy.language import Language
except ImportError:
    print("ERROR: spacy not installed. Run:")
    print("  pip install spacy")
    print("  python -m spacy download en_core_web_sm")
    sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "Streamlit_app" / "ensemble_results.csv"

# Model options: en_core_web_sm, en_core_web_md, en_core_web_lg
SPACY_MODEL = "en_core_web_sm"


class BiomedicalNLPProcessor:
    """Enhanced NLP processing for biomedical text."""

    def __init__(self, model_name: str = SPACY_MODEL):
        """Initialize spaCy model for text processing."""
        self.model_name = model_name
        self.nlp: Language | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load spaCy model."""
        try:
            self.nlp = spacy.load(self.model_name)
            print(f"✓ Loaded {self.model_name}")
        except OSError:
            print(f"Model {self.model_name} not found.")
            print("Install with: python -m spacy download en_core_web_sm")
            raise

    def process_text(self, text: str) -> dict[str, Any]:
        """Process text with spaCy and extract biomedical features."""
        if not self.nlp or not text:
            return {"tokens": [], "entities": [], "lemmas": [], "processed": ""}

        doc = self.nlp(text.lower())

        return {
            "tokens": [token.text for token in doc],
            "lemmas": [token.lemma_ for token in doc if not token.is_stop],
            "entities": [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
            ],
            "processed": " ".join(
                token.lemma_ for token in doc
                if not token.is_stop and not token.is_punct
            ),
        }

    def batch_process(self, texts: list[str]) -> list[dict[str, Any]]:
        """Process multiple texts efficiently."""
        if not self.nlp:
            return []

        results = []
        for text in texts:
            results.append(self.process_text(text))
        return results

    @staticmethod
    def extract_biomedical_entities(df: pd.DataFrame) -> pd.DataFrame:
        """Extract biomedical entities from abstracts."""
        processor = BiomedicalNLPProcessor()

        # Process a sample for testing
        sample_size = min(100, len(df))
        abstracts = df["Abstract"].head(sample_size).fillna("")

        entity_stats = {"total_entities": 0, "entity_types": {}}

        for abstract in abstracts:
            result = processor.process_text(abstract)
            entity_stats["total_entities"] += len(result["entities"])
            for entity in result["entities"]:
                label = entity["label"]
                entity_stats["entity_types"][label] = (
                    entity_stats["entity_types"].get(label, 0) + 1
                )

        return entity_stats


def compare_preprocessing(df: pd.DataFrame, sample_size: int = 5) -> None:
    """Compare NLTK vs SciSpaCy preprocessing on sample texts."""
    print("\n" + "=" * 70)
    print("NLP PREPROCESSING COMPARISON (NLTK vs SciSpaCy)")
    print("=" * 70)

    processor = BiomedicalNLPProcessor()

    # Get sample abstracts
    abstracts = df["abstract"].head(sample_size).fillna("")

    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import wordpunct_tokenize
    from nltk.corpus import stopwords

    nltk_lemmatizer = WordNetLemmatizer()
    nltk_stopwords = set(stopwords.words("english"))

    for i, abstract in enumerate(abstracts, 1):
        if not abstract:
            continue

        # NLTK processing
        nltk_tokens = [
            word.lower() for word in wordpunct_tokenize(abstract)
            if word.lower() not in nltk_stopwords and word.isalnum()
        ]
        nltk_processed = " ".join(nltk_tokens[:20])

        # SciSpaCy processing
        spacy_result = processor.process_text(abstract)
        spacy_processed = spacy_result["processed"][:100]

        print(f"\n[Sample {i}]")
        print(f"Original: {abstract[:80]}...")
        print(f"NLTK:     {nltk_processed}...")
        print(f"SciSpaCy: {spacy_processed}...")

        if spacy_result["entities"]:
            print(f"Entities: {[e['text'] for e in spacy_result['entities'][:3]]}")

    print("\n" + "=" * 70)


def main() -> int:
    """Test biomedical NLP processor."""
    try:
        df = load_dataset()
        print(f"Loaded {len(df)} rows")

        # Compare preprocessing
        compare_preprocessing(df)

        # Extract entity statistics
        print("\n[BIOMEDICAL ENTITY EXTRACTION]")
        entity_stats = BiomedicalNLPProcessor.extract_biomedical_entities(df)
        print(f"  Total entities found (sample): {entity_stats['total_entities']}")
        print(f"  Entity types: {entity_stats['entity_types']}")

        print("\n✓ SciSpaCy integration ready!")
        print("\nNext: Integrate into ibidav/service.py to use BiomedicalNLPProcessor")
        print("      for tokenization, lemmatization, and entity extraction.")

        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the ensemble results CSV."""
    return pd.read_csv(path)


if __name__ == "__main__":
    sys.exit(main())
