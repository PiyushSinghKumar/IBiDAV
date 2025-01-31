# IBiDAV

## Overview
IBiDAV (Intelligent Biomedical Data Analysis and Visualization) is a Streamlit-based application designed for processing and analyzing biomedical text data. It features text preprocessing, word cloud visualization, and a topic modeling system using Latent Dirichlet Allocation (LDA). Users can search and filter articles based on metadata and keywords.

## Features
- **Text Preprocessing**: Converts text to lowercase, tokenizes, removes non-alphabetic words, digits, stopwords, and lemmatizes the text.
- **Word Cloud Generation**: Creates a word cloud from the processed text data.
- **LDA Topic Modeling**: Extracts key topics from biomedical text using Latent Dirichlet Allocation.
- **Search Functionality**: Allows users to search for articles using PMID, PMCID, Title, or Abstract.
- **Category-Based Filtering**: Displays relevant biomedical papers and associated images.
- **Dynamic Content Loading**: Supports incremental loading of search results for better performance.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Required Python libraries: `nltk`, `pandas`, `streamlit`, `matplotlib`, `wordcloud`, `gensim`

### Setup
1. Clone the repository:
   ```bash
   git clone
   cd ibidav
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Upload a CSV file containing biomedical text data.
3. Use the search bar to filter articles by PMID, PMCID, Title, or Abstract.
4. View word clouds and topic modeling results.
5. Browse articles and view associated images.

## File Structure
```
IBiDAV/
│── app.py               # Main application script
│── requirements.txt     # List of dependencies
│── data/
│   ├── ensemble_results.csv  # Sample dataset
│── utils/
│   ├── preprocess.py    # Text preprocessing functions
│   ├── visualization.py # Word cloud and LDA functions
```

## Functions
### `preprocess_text(text: str) -> str`
Preprocesses input text by:
- Lowercasing
- Tokenizing
- Removing non-alphabetic words and digits
- Lemmatizing
- Removing stopwords

### `generate_wordcloud(top_words_text: str) -> None`
Creates and displays a word cloud using `WordCloud` from the given text.

### `load_and_preprocess_data(file_path: str) -> pd.DataFrame`
Loads and preprocesses data from a CSV file, merging relevant columns into a 'Corpus'.

### `build_lda_model(processed_corpus: pd.Series, num_topics: int) -> LdaModel`
Builds an LDA topic model from the processed corpus.

### `main_page() -> None`
Displays the main application interface, including:
- Word cloud visualization
- Search bar for article filtering

### `display_filtered_data(category: str) -> None`
Filters and displays articles by category, including images.
