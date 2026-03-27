"""Label classification trainer for multi-label prediction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / ".cache" / "models"


class LabelClassifier:
    """Multi-label classification for auto-labeling articles."""

    def __init__(self):
        """Initialize classifier components."""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
        )
        self.mlb = MultiLabelBinarizer()
        self.classifier: MultiOutputClassifier[Any] | None = None
        self._is_trained = False

    def _parse_labels(self, label_str: str) -> list[str]:
        """Parse comma/semicolon separated labels."""
        if not isinstance(label_str, str) or not label_str.strip():
            return []
        # Handle both comma and semicolon separators
        labels = [l.strip() for l in label_str.replace(";", ",").split(",")]
        return [l for l in labels if l]

    def train(
        self,
        texts: list[str],
        labels: list[str],
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """
        Train classifier on labeled data.

        Args:
            texts: List of input texts (titles + abstracts)
            labels: List of label strings (comma-separated)
            test_size: Fraction for validation

        Returns:
            Training metrics
        """
        # Parse multi-labels
        parsed_labels = [self._parse_labels(lbl) for lbl in labels]

        # Filter out samples with no labels
        filtered_texts = []
        filtered_labels = []
        for text, label_list in zip(texts, parsed_labels):
            if label_list:  # Only keep labeled samples
                filtered_texts.append(text)
                filtered_labels.append(label_list)

        if not filtered_texts:
            return {"error": "No labeled samples found"}

        # Vectorize texts
        X = self.vectorizer.fit_transform(filtered_texts)

        # Binarize labels
        y = self.mlb.fit_transform(filtered_labels)

        # Train multi-output logistic regression
        base_classifier = LogisticRegression(
            max_iter=500,
            n_jobs=-1,
            random_state=42,
        )
        self.classifier = MultiOutputClassifier(base_classifier)
        self.classifier.fit(X, y)

        self._is_trained = True

        # Calculate training accuracy
        y_pred = self.classifier.predict(X)
        correct = sum(1 for pred, true in zip(y_pred, y) if (pred == true).all())
        accuracy = correct / len(y) if y.shape[0] > 0 else 0.0

        return {
            "trained_samples": len(filtered_texts),
            "unique_labels": len(self.mlb.classes_),
            "label_classes": list(self.mlb.classes_),
            "accuracy": round(accuracy, 4),
            "feature_count": X.shape[1],
        }

    def predict(self, text: str, top_k: int = 3) -> list[str]:
        """
        Predict labels for new text.

        Args:
            text: Input text
            top_k: Return top-k predictions

        Returns:
            List of predicted labels
        """
        if not self._is_trained or not self.classifier:
            return []

        try:
            X = self.vectorizer.transform([text])
            # Get predictions from each binary classifier
            predictions = []
            for i, est in enumerate(self.classifier.estimators_):
                prob = est.predict_proba(X)[0, 1]  # Probability of positive class
                label = self.mlb.classes_[i]
                predictions.append((label, prob))

            # Sort by confidence and return top-k
            predictions.sort(key=lambda x: x[1], reverse=True)
            return [label for label, _ in predictions[:top_k]]
        except Exception:
            return []

    def batch_predict(self, texts: list[str]) -> list[list[str]]:
        """Predict labels for multiple texts."""
        return [self.predict(text) for text in texts]

    def save(self, path: Path | None = None) -> Path:
        """Save classifier to disk."""
        path = path or (MODEL_DIR / "label_classifier.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "mlb": self.mlb,
                "classifier": self.classifier,
                "is_trained": self._is_trained,
            }, f)

        return path

    @classmethod
    def load(cls, path: Path | None = None) -> LabelClassifier:
        """Load classifier from disk."""
        path = path or (MODEL_DIR / "label_classifier.pkl")

        if not path.exists():
            return cls()

        try:
            with path.open("rb") as f:
                state = pickle.load(f)

            instance = cls()
            instance.vectorizer = state["vectorizer"]
            instance.mlb = state["mlb"]
            instance.classifier = state["classifier"]
            instance._is_trained = state["is_trained"]
            return instance
        except Exception:
            return cls()
