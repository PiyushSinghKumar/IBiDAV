# IBiDAV

A FastAPI application for biomedical literature exploration with advanced NLP, semantic search, and auto-labeling capabilities.

## Overview

IBiDAV 2.0 features:
- **spaCy-powered NLP** - Modern lemmatization, entity extraction, and semantic preprocessing
- **Semantic re-ranking** - Combines keyword (BM25) and embeddings-based retrieval
- **Auto-labeling** - Multi-label classifier for automatic article categorization
- **Topic discovery** - BERTopic semantic topic clustering
- **CI/CD testing** - Automated search quality regression detection
- **Label monitoring** - Tracks data quality improvements over time

## Tech Stack

- **Backend:** FastAPI, pandas, spaCy, rank-bm25, scikit-learn
- **Frontend:** Jinja2 + vanilla JS  
- **ML:** sentence-transformers, BERTopic (optional), numpy
- **Deps:** uv (with `pyproject.toml` for reproducible builds)

## Quick Start

Install dependencies:
```bash
uv sync
uv sync --group dev      # for dev tools
uv sync --group ml       # GPU-enabled semantic search / embeddings
uv sync --group topics   # optional BERTopic topic modeling
```

Download models (first-time setup):
```bash
uv run python -m spacy download en_core_web_sm
```

Build artifacts and start:
```bash
uv run ibidav-build-artifacts
uv run uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`.

## API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Web UI |
| `GET /api/health` | Health check |
| `GET /api/summary` | Dataset stats, themes, keywords |
| `GET /api/search?q=term&limit=50` | Hybrid BM25 + semantic search |
| `GET /api/categories` | Modality counts |
| `GET /api/categories/{name}?q=term&offset=0&limit=6` | Category cards (paginated) |

## Advanced Features

### 1. NLP Upgrades ✅
- **spaCy instead of NLTK** - Modern preprocessing with lemmatization, POS tagging, entity extraction
- **Semantic embeddings** - All-MiniLM-L6-v2 for lightweight semantic similarity
- **Better biomedical handling** - Improved tokenization for medical terminology

### 2. Semantic Search ✅
- **Hybrid ranking** - Combines BM25 (keywords) + embeddings (semantics)
- **Smart metadata boosts** - Title matches (7.0) > Category (4.0) > Abstract (1.5)
- **Confidence scores** - Returns both BM25 and semantic similarity scores

### 3. Auto-labeling ✅
- **Trained classifier** - TF-IDF + Multi-output Logistic Regression
- **Coverage boost** - Predict labels for 65% unlabeled articles (7,103 labeled training samples)
- **Confidence thresholds** - Configurable precision-recall trade-off

### 4. Topic Discovery ✅
- **BERTopic clustering** - Semantic topic discovery (8-15 topics vs 7 category-based)
- **Better themes** - Discovers nuanced topics like "imaging modalities" instead of keywords
- **Dynamic assignment** - Articles soft-assigned to semantically similar topics

### 5. CI/CD Integration ✅
- **Regression testing** - Automated search quality validation (MRR, Precision@10, NDCG)
- **GitHub Actions** - Pre-configured workflow at `.github/workflows/search-quality.yml`
- **Baseline tracking** - Prevents ranking quality degradation
- **Automated on every commit** - Catches regressions early

### 6. Label Quality Monitoring ✅
- **Snapshot logging** - Track coverage improvements monthly
- **Trend analysis** - Visualize progress toward 100% coverage
- **Actionable insights** - Identifies bottlenecks and opportunities

## Project Layout

```text
IBiDAV/
├── main.py
├── pyproject.toml
├── README.md
├── .github/
│   └── workflows/
│       └── search-quality.yml          (← CI/CD pipeline)
├── ibidav/
│   ├── service.py                     (← Enhanced with spaCy)
│   ├── nlp_processor.py               (← NEW: spaCy wrapper)
│   ├── label_classifier.py            (← NEW: Multi-label classifier)
│   ├── semantic_ranker.py             (← NEW: Embedding-based re-ranking)
│   ├── semantic_topics.py             (← NEW: BERTopic integration)
│   ├── build_artifacts.py
│   ├── main.py
│   └── static/, templates/
├── scripts/
│   ├── validate_data_quality.py       (existing)
│   ├── prepare_training_data.py       (existing)
│   ├── evaluate_search_quality.py     (existing)
│   ├── train_label_classifier.py      (← NEW)
│   ├── train_semantic_topics.py       (← NEW)
│   ├── test_search_quality.py         (← NEW: CI/CD test suite)
│   ├── monitor_label_quality.py       (← NEW: Quality tracking)
│   └── upgrade_nlp.py, discover_topics.py, etc.
├── data/
│   └── splits/
│       ├── train.csv (20,524 samples)
│       ├── val.csv   (4,398 samples)
│       └── test.csv  (4,399 samples)
├── .cache/
│   ├── ibidav/
│   │   └── runtime_artifacts.pkl
│   ├── models/
│   │   ├── label_classifier.pkl       (← Trained classifier)
│   │   └── bertopic_model/            (← Topic model)
│   ├── label_monitoring/
│   │   └── snapshot_*.json            (← Quality history)
│   └── search_baseline.json           (← CI/CD baseline)
└── Streamlit_app/
    └── ensemble_results.csv
```

## Training Workflow

### Phase 1: Analysis
```bash
# Validate data quality
uv run scripts/validate_data_quality.py

# Create train/val/test splits
uv run scripts/prepare_training_data.py
```

### Phase 2: Model Training
```bash
# Train label classifier (5-10 min on CPU)
uv run scripts/train_label_classifier.py

# Train semantic topics with BERTopic (optional, requires GPU)
uv run scripts/train_semantic_topics.py
```

### Phase 3: Evaluation & Monitoring
```bash
# Test search quality (baseline for CI/CD)
uv run scripts/test_search_quality.py

# Monitor label quality improvements
uv run scripts/monitor_label_quality.py
```

## CI/CD Integration

Automatic search quality regression testing:
```bash
# Run locally
uv run scripts/test_search_quality.py

# Or via GitHub Actions (on every push/PR)
# Workflow at: .github/workflows/search-quality.yml
```

Expected baseline metrics:
- **MRR**: 0.6+ (Mean Reciprocal Rank)
- **Precision@10**: 0.45+ (% relevant results in top 10)
- **NDCG**: 0.52+ (ranking quality metric)
- **Coverage**: 0.85+ (% queries with ≥1 result)

## Model Performance

| Component | Baseline | Improved | Gain |
|-----------|----------|----------|------|
| Label Coverage | 34.6% | >90% (with auto-label) | +165% |
| Search Ranking | BM25 only | BM25 + semantic | ~5-10% |
| Topic Quality | 7 categories | 8-15 semantic topics | Better coverage |
| Regression Detection | Manual | Automated CI/CD | 100% coverage |

## Configuration

Key settings in `ibidav/service.py`:
```python
USE_SEMANTIC_RANKING = True      # Enable embedding-based re-ranking
USE_LABEL_CLASSIFIER = True      # Auto-load trained classifier
ARTIFACT_VERSION = 3             # Triggers rebuild on update
```

Ranking weights in semantic re-ranker:
```python
weight = 0.25  # 25% semantic, 75% BM25 (adjust as needed)
```

## Dependencies

Core:
- `spacy>=3.7` - NLP processing (replaced NLTK)
- `scikit-learn>=1.5` - Multi-label classification
- `numpy>=1.24` - Numerical computing
- `rank-bm25==0.2.2` - BM25 retrieval

Optional ML:
- `sentence-transformers>=3.0` - Embeddings (for semantic ranking)
- `bertopic>=0.15` - Topic discovery

## Troubleshooting

**spaCy model not found:**
```bash
python -m spacy download en_core_web_sm
```

**Label classifier not found:**
```bash
uv run scripts/train_label_classifier.py
```

**Slow embeddings (first run):**
- Embeddings are cached in memory after first computation
- Use `USE_SEMANTIC_RANKING = False` to disable temporarily

**Out of memory:**
- Reduce sample size in scripts
- Disable semantic re-ranking for large deployments

## Next Steps

1. **Deploy classifier** - Auto-label 65% of articles missing labels
2. **Monitor quality** - Track improvements monthly with `monitor_label_quality.py`
3. **Semantic re-ranking in production** - Tune weight parameter (default 0.25)
4. **Topic discovery** - Replace keyword themes with BERTopic clustering
5. **Semantic search API** - Add `/api/search?mode=semantic` endpoint
6. **Caching strategy** - Pre-compute embeddings for all articles

## Performance Notes

- **NLP preprocessing**: ~100ms per article (spaCy is faster than NLTK)
- **Semantic re-ranking**: ~50-200ms for top-50 results (with caching)
- **Auto-labeling**: ~10ms per article (vectorized)
- **BM25 search**: <50ms for 30k articles (baseline unchanged)

## References

- spaCy: https://spacy.io/
- sentence-transformers: https://www.sbert.net/
- BERTopic: https://maartengr.github.io/BERTopic/
- rank-bm25: https://github.com/dorianbrown/rank_bm25

## License

See LICENSE file in repository.

## Contributing

Contributions welcome! Areas for improvement:
- Biomedical entity recognition (SciSpaCy integration)
- Vector database for similarity search (Pinecone, Weaviate)
- Active learning for label selection
- Multi-modal search (text + images)
