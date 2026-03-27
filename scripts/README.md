# IBiDAV Training & Evaluation Scripts

This directory contains tools for improving model quality, validating data, and evaluating search performance.

## Installation

Install ML dependencies:

```bash
uv sync --group ml
uv sync --group topics  # optional BERTopic dependencies
```

## Scripts Overview

### 1. Validate Data Quality
**File:** `validate_data_quality.py`

Analyzes multi-label coverage, identifies missing labels, and provides curation recommendations.

```bash
uv run scripts/validate_data_quality.py
```

**Output:**
- Label distribution analysis
- Text field completeness (title, abstract, PMID, PMCID)
- Coverage percentage and recommendations

---

### 2. Upgrade NLP to SciSpaCy
**File:** `upgrade_nlp.py`

Replaces NLTK with biomedical-aware NLP using scispacy. Performs:
- Named entity recognition for biomedical terms
- Biomedical lemmatization
- Domain-specific stopword handling

```bash
uv run scripts/upgrade_nlp.py
```

**Comparison:**
- Compares NLTK vs SciSpaCy preprocessing on samples
- Extracts biomedical entity statistics
- Guides integration into `ibidav/service.py`

**Integration Steps:**
1. Install: `uv pip install scispacy`
2. Download model: `python -m spacy download en_core_sci_sm`
3. Replace `NLTK` preprocessing in `ibidav/service.py`

---

### 3. Discover Topics with Embeddings
**File:** `discover_topics.py`

Implements BERTopic for semantic topic discovery, improving over keyword-based extraction.

```bash
uv run scripts/discover_topics.py
```

**Methods Compared:**
- **BERTopic**: Embedding-based semantic topics with coherence scores
- **Keyword**: Current heuristic keyword extraction

**Features:**
- Uses `allenai-specter` embeddings (science-aware transformer)
- Automatic topic coherence estimation
- Scalable to large document collections

---

### 4. Evaluate Search Quality
**File:** `evaluate_search_quality.py`

Offline evaluation framework for BM25 ranking without manual relevance labels.

```bash
uv run scripts/evaluate_search_quality.py
```

**Metrics:**
- Mean Reciprocal Rank (MRR)
- Precision@10
- NDCG (Normalized Discounted Cumulative Gain)
- Coverage (% queries with ≥1 relevant result)

**Features:**
- Synthetic query generation from document corpus
- Automatic relevance inference from term overlap
- Baseline for improving ranking

---

### 5. Prepare Training Data
**File:** `prepare_training_data.py`

Creates train/val/test splits (70/15/15) with reproducible random seed.

```bash
uv run scripts/prepare_training_data.py
```

**Output:**
- `data/splits/train.csv` — Training set
- `data/splits/val.csv` — Validation set
- `data/splits/test.csv` — Test set
- `data/splits/split_metadata.json` — Reproducibility metadata

**Use Cases:**
- Training label classifiers
- Fine-tuning topic models
- Learning-to-rank models
- Hyperparameter optimization on val set

---

## Recommended Workflow

### Phase 1: Analysis & Baseline
1. Run `validate_data_quality.py` to understand label sparsity
2. Run `evaluate_search_quality.py` for baseline BM25 metrics
3. Run `prepare_training_data.py` to create splits

### Phase 2: Model Improvements
1. Run `upgrade_nlp.py` and integrate SciSpaCy
2. Run `discover_topics.py` and integrate BERTopic
3. Retrain BM25 index with improved tokenization

### Phase 3: Evaluation
1. Re-run `evaluate_search_quality.py` on improved ranking
2. Measure impact on relevance metrics
3. Add semantic re-ranking with embeddings

---

## Integration Examples

### Adding SciSpaCy to Service

```python
# In ibidav/service.py

from scripts.upgrade_nlp import BiomedicalNLPProcessor

processor = BiomedicalNLPProcessor("en_core_sci_sm")
result = processor.process_text(abstract)
processed_text = result["processed"]
extracted_entities = result["entities"]
```

### Using Training Splits

```python
import pandas as pd

train_df = pd.read_csv("data/splits/train.csv")
val_df = pd.read_csv("data/splits/val.csv")

# Train label classifier
# clf = train(train_df["title"], train_df["multi_labels"])
# eval_score = evaluate(clf, val_df["title"], val_df["multi_labels"])
```

### Deploying Topic Discovery

```python
from scripts.discover_topics import EmbeddingTopicDiscovery

discoverer = EmbeddingTopicDiscovery(num_topics=15)
topics, topic_info = discoverer.discover_topics(abstracts)

# Replace _build_topics() in service.py
# Cache in runtime_artifacts.pkl
```

---

## Architecture & Design

| Component | Current | Proposed |
|-----------|---------|----------|
| **Tokenization** | NLTK wordpunct | SciSpaCy lemmatizer |
| **Stopwords** | NLTK English | SciSpaCy (biomedical-aware) |
| **Topics** | Keyword frequency | BERTopic embeddings |
| **Search Ranking** | BM25 | BM25 + semantic re-rank |
| **Evaluation** | Manual inspection | Automated NDCG/MRR |
| **Data Splits** | Ad-hoc | Reproducible train/val/test |

---

## Next Steps

1. **Integrate improvements** into `ibidav/service.py` and `build_artifacts.py`
2. **Add CI/CD testing** using `evaluate_search_quality.py`
3. **Implement label classifier** using `data/splits/train.csv`
4. **Add semantic re-ranking** with sentence embeddings
5. **Monitor** label quality improvements over time

---

## Requirements

- Python ≥3.11
- `uv` package manager
- NLTK corpora: `python -m nltk.downloader stopwords wordnet`

See `pyproject.toml` dependency-groups for version pinning.
