import os
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import streamlit as st
import math
import re
from collections import Counter

st.markdown("""
<style>
            .css-zq5wmm.ezrtsby0
            {
            visibility:hidden;
            }
            .css-cio0dv.ea3mdgi1
            {
            visibility:hidden;
            }
            .css-10trblm.e1nzilvr0
            {
            display: flex;
            justify-content: center;
            align-items: center;
            }
            </style>
""", unsafe_allow_html=True)


def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by lowercasing, tokenizing, removing non-alphabetic words,
    removing digits, lemmatizing, and removing stopwords.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    if isinstance(text, str):
        text = text.lower()
        words = nltk.word_tokenize(text)
        words = [word for word in words if word.isalpha()]
        words = [re.sub(r'[^A-Za-z0-9\s]', '', word) for word in words]
        words = [word for word in words if not word.isdigit()]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)
    else:
        return ''


def generate_wordcloud(top_words_text: str) -> None:
    """
    Generate and display a word cloud from the given text.

    Args:
        top_words_text (str): The text to create a word cloud from.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='black', prefer_horizontal=1.0,
                          stopwords=set(STOPWORDS)).generate(top_words_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


@st.cache_data
def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file, combine relevant columns into a 'Corpus', and preprocess the text.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data.
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, file_path)
    df = pd.read_csv(file_path)
    df['Corpus'] = df['Title'] + ' ' + df['Abstract'] + ' ' + df['Category']
    df['Processed_Corpus'] = df['Corpus'].apply(preprocess_text)
    return df


@st.cache_data
def build_lda_model(processed_corpus: pd.Series, num_topics: int = 10) -> LdaModel:
    """
    Build an LDA model using the processed text corpus.

    Args:
        processed_corpus (pd.Series): The processed corpus.
        num_topics (int): The number of topics for the LDA model.

    Returns:
        LdaModel: The trained LDA model.
    """
    tokenized_corpus = processed_corpus.apply(nltk.word_tokenize)
    dictionary = Dictionary(tokenized_corpus)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_corpus]
    lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model


def main_page() -> None:
    """
    The main page of the application, displaying word cloud and search functionality.
    """
    df = load_and_preprocess_data("ensemble_results.csv")
    word_frequencies = Counter()

    num_topics = 10
    lda_model = build_lda_model(df['Processed_Corpus'], num_topics=num_topics)

    top_words_text = ''
    for topic_num in range(num_topics):
        topic_words = lda_model.show_topic(topic_num, topn=10)
        top_words = [word for word, _ in topic_words if len(word) > 3 and not word.endswith("ing") and not word.endswith("image")]
        top_words_text += ' '.join(top_words) + ' '

    for text in df['Processed_Corpus']:
        words = text.split()
        for word in words:
            if word in top_words_text:
                word_frequencies[word] += 1

    st.title('IBiDAV')

    search_query = st.text_input('Enter your search query:', placeholder='Type PMID / PMCID / Title')
    search_button = st.button('Search')

    st.write("---")
    if search_button or search_query:
        if search_query.strip() == "":
            st.warning("Please enter query")
        else:
            if search_button or search_query:
                words = search_query.split('+')
                pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word.strip()) for word in words) + r')\b', flags=re.IGNORECASE)

                pmid_match = df['PMID'].astype(str).apply(lambda x: bool(pattern.search(x)))
                pmcid_match = df['PMCID'].astype(str).apply(lambda x: bool(pattern.search(x)))
                title_match = df['Title'].apply(lambda x: bool(pattern.search(x)))
                df['Abstract'] = df['Abstract'].apply(lambda x: str(x))
                abstract_match = df['Abstract'].apply(lambda x: bool(pattern.search(x)))

                filtered_df = df[pmid_match | pmcid_match | title_match | abstract_match]
                filtered_df = filtered_df.drop_duplicates(subset=['PMCID', 'PMID', 'Title', 'Abstract', 'Article URL'])

                st.dataframe(filtered_df[['PMCID', 'PMID', 'Title', 'Abstract', 'Article URL']], use_container_width=True, hide_index=True)
                st.empty()

    if not (search_button or search_query):
        generate_wordcloud(top_words_text)


def display_filtered_data(category: str) -> None:
    """
    Display filtered data for a specific category.

    Args:
        category (str): The category to filter data by.
    """
    df = load_and_preprocess_data("ensemble_results.csv")
    df['Abstract'] = df['Abstract'].apply(lambda x: str(x))

    filtered_df = df[(df['Category'] == category) & (df['multi_labels'] == category)]
    filtered_df = filtered_df.groupby(['PMCID', 'PMID', 'Title', 'Abstract', 'Article URL']).agg({
        'Image URL': lambda x: ', '.join(x)
    }).reset_index()
    filtered_df.rename(columns={'Image URL': 'Image URLs'}, inplace=True)

    filtered_df = filtered_df.drop_duplicates(subset=['PMCID', 'PMID', 'Title', 'Abstract', 'Article URL'])
    st.title(category)

    search_query = st.text_input('Enter your search query:', placeholder='Type PMID / PMCID / Title')
    search_button = st.button("Search")

    if search_query == "":
        if hasattr(st.session_state, 'applied_search_query'):
            del st.session_state.applied_search_query

    if search_button or search_query:
        st.session_state.applied_search_query = search_query

    if hasattr(st.session_state, 'applied_search_query') and st.session_state.applied_search_query:
        search_query = st.session_state.applied_search_query
        words = search_query.split('+')
        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word.strip()) for word in words) + r')\b', flags=re.IGNORECASE)

        pmid_match = filtered_df['PMID'].astype(str).apply(lambda x: bool(pattern.search(x)))
        pmcid_match = filtered_df['PMCID'].astype(str).apply(lambda x: bool(pattern.search(x)))
        title_match = filtered_df['Title'].apply(lambda x: bool(pattern.search(x)))
        abstract_match = filtered_df['Abstract'].apply(lambda x: bool(pattern.search(x)))

        filtered_df = filtered_df[pmid_match | pmcid_match | title_match | abstract_match]

    pmids = list(filtered_df['PMID'])
    pmcids = list(filtered_df['PMCID'])
    titles = list(filtered_df['Title'])
    abstracts = list(filtered_df['Abstract'])
    urls = list(filtered_df['Article URL'])
    iurls = list(filtered_df['Image URLs'])

    num_initial_papers = 6
    num_papers_per_load = 3

    if 'num_displayed_papers' not in st.session_state:
        st.session_state.num_displayed_papers = num_initial_papers

    container = st.container()

    for i in range(st.session_state.num_displayed_papers):
        if i < len(pmids) and i < len(pmcids):
            with container:
                card = st.container()
                with card:
                    col1, col2 = st.columns([6, 1])

                    with col1:
                        with st.container():
                            st.write("---")
                            st.write(f"PMID: {pmids[i]} , PMCID: {pmcids[i]}")
                            st.write(f"Title: {titles[i]}")

                            abstract = abstracts[i]
                            if isinstance(abstract, str):
                                abstract = abstract[:100] + "..." if len(abstract) > 100 else abstract
                                st.write(f"Abstract: {abstract[:100]}...")
                            else:
                                st.write(f"Abstract: Not Available")
                            st.write(f"Article URL: {urls[i]}")

                    with col2:
                        with st.container():
                            st.write("---")
                            image_urls = iurls[i].split(',')
                            num_images = min(len(image_urls), 6)
                            num_columns = 3
                            num_rows = int(math.ceil(num_images / num_columns))

                            for row in range(num_rows):
                                with st.container():
                                    for col in range(num_columns):
                                        image_index = row * num_columns + col
                                        if image_index < num_images:
                                            st.image(image_urls[image_index].strip(), width=50)

                card.markdown(
                    f"""<style>
                        .reportview-container .main .block-container {{
                            width: 100%;
                            height: auto;
                        }}
                    </style>""",
                    unsafe_allow_html=True
                )
    st.markdown(
        f"Displaying {st.session_state.num_displayed_papers} out of {len(pmids)} results.",
        unsafe_allow_html=True
    )

    if st.session_state.num_displayed_papers < len(pmids):
        if st.button("Load More"):
            st.session_state.num_displayed_papers += num_papers_per_load
    else:
        st.write("All papers have been loaded.")

def page2():

    display_filtered_data("Computed Tomography (CT) Scan")

def page3():

    display_filtered_data("Endoscopy")

def page4():

    display_filtered_data("Histology")

def page5():

    display_filtered_data("Magnetic Resonance Imaging (MRI)")

def page6():

    display_filtered_data("Positron Emission Tomography (PET) Scan")

def page7():

    display_filtered_data("Ultrasound")

def page8():

    display_filtered_data("X-ray")

def page9():
    st.title('IBiDAV')
    st.write("""
            Welcome to IBiDAV : Integrative Biomedical Data Analysis and Visualization.

            A project designed to revolutionize the way we explore and analyze biomedical data.
            IBiDAV is a dynamic platform that seamlessly integrates cutting-edge image classification and natural language processing.

            Dive into a vast repository of medical images and scholarly literature, categorize papers, uncover hidden themes, and conduct intricate searches.
            Our intuitive interface serves as a bridge between image and text analysis, providing a comprehensive understanding of the biomedical domain.

            Join us on this transformative journey through the intricate landscape of biomedical research.
            Let IBiDAV empower you to discover, learn, and innovate in the world of healthcare and life sciences.
             """)
    st.write("""
    Developed by - Piyush Kumar Singh

    Guided by - Prof. Soldatos, Theodoros
    """)

page_names_to_funcs = {
    "Home": main_page,
    "Computed Tomography (CT) Scan": page2,
    "Endoscopy": page3,
    "Histology": page4,
    "Magnetic Resonance Imaging (MRI)": page5,
    "Positron Emission Tomography (PET) Scan": page6,
    "Ultrasound": page7,
    "X-ray": page8,
    "About": page9
}

selected_page = "Home"


if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Home"


for page_name in page_names_to_funcs.keys():
    if st.sidebar.button(page_name, use_container_width=True):

        if hasattr(st.session_state, 'applied_search_query'):
            del st.session_state.applied_search_query
        st.session_state.selected_page = page_name
        st.session_state.num_displayed_papers = 6


page_names_to_funcs[st.session_state.selected_page]()
