import ast
import io
import json
import math
import re
from typing import NoReturn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models
import seaborn as sns
from wordcloud import WordCloud
import spacy
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from pywebio.input import select, input, TEXT
from pywebio.output import put_text, put_image, put_html
from pywebio.platform.flask import webio_view
from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects


from flask import Flask

app = Flask(__name__)


# TMDB developer API keys are available from TMDB: https://developers.themoviedb.org/3/getting-started/introduction

def tmdb_keyword_query(kw_list: list, api_key: str) -> list:
    """
    Uses the TMDB Web API to generate list of movie ids based on keyword search.
    :param kw_list: list of keyword strings
    :param api_key: API key string
    :return: list of unique ids
    """

    # return keyword ids and store them as a pipe-delimited string

    kw_id_list = []
    kw_list = list(kw_list.split(','))
    put_text("Finding movies that match keyword(s) {}...".format(str(kw_list).replace('[', '')
                                                                 .replace(']', '')))

    for kw in kw_list:
        url = 'https://api.themoviedb.org/3/search/keyword?api_key={key}&query={query}' \
            .format(key=api_key, query=kw)

        session = Session()

        try:
            response = session.get(url)
            kw_ids = json.loads(response.text)

        except (ConnectionError, Timeout, TooManyRedirects) as e:
            print(e)

        df = pd.json_normalize(kw_ids['results'])
        kw_ids = list(df['id'])
        kw_ids = [str(kw_id) for kw_id in kw_ids]
        for kw_id in kw_ids:
            kw_id_list.append(kw_id)

    piped_ids = '|'.join(kw_id_list)

    # return first page of results, check total page count for all results, and set page range maximum

    url = '''https://api.themoviedb.org/3/discover/movie?with_keywords={ids}
            &page={page_num}
            &api_key={key}&'''.format(ids=piped_ids, page_num=1, key=api_key)

    session = Session()

    try:
        response = session.get(url)
        kw_data = json.loads(response.text)
        page_max = kw_data['total_pages'] + 1

    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)

    # loop through page range to generate list of movie ids

    id_list = []
    for p in range(1, page_max):
        url = '''https://api.themoviedb.org/3/discover/movie?with_keywords={ids}
            &page={page_num}
            &api_key={key}&'''.format(ids=piped_ids, page_num=p, key=api_key)

        session = Session()

        try:
            response = session.get(url)
            kw_data = json.loads(response.text)
        except (ConnectionError, Timeout, TooManyRedirects) as e:
            print(e)

        kw_df = pd.json_normalize(kw_data['results'])
        id_list.append(kw_df['id'])

    id_list = [i_list for i in id_list for i_list in i]

    # estimate time to build dataset

    time_estimate = math.ceil(len(id_list) * 0.002)

    put_text(
        '{} movies found! It will take about {} minute(s) to build a dataset based on these results.'.format(
            f'{len(id_list):,d}', time_estimate)
    )
    return id_list


def build_dataset(id_list: list, api_key: str) -> pd.DataFrame:
    """
    Builds dataset using TMDB web API
    :param id_list: list of TMDB unique ids
    :param api_key:
    :return: Pandas DataFrame
    """

    data_dict = {}
    for id in id_list:
        url = 'https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}'.format(movie_id=id, key=api_key)

        session = Session()

        try:
            response = session.get(url)
            data = json.loads(response.text)

        except (ConnectionError, Timeout, TooManyRedirects) as e:
            print(e)

        data_dict[id] = data

    data_df = pd.DataFrame.from_dict(data_dict, orient='index')
    put_text('Dataset successfully created! ({} rows, {} columns)'.format(
        data_df.shape[0], data_df.shape[1]))
    return data_df


def transform_categories(value_list: list) -> str:
    """
    Transforms categorical data (e.g. genres, production companies, production countries) into parsable strings.
    :param value_list: list of dicts
    :return: string with comma-delimited categories
    """
    string = str(value_list)
    dicts_list = ast.literal_eval(string)
    string_list = []
    for d in dicts_list:
        string_list.append(d['name'])
    new_string = ','.join(string_list)

    return new_string


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data and returns selected columns.
    :param data: Pandas DataFrame
    :return: Pandas DataFrame, cleaned
    """
    df_releases = data.dropna(subset=['release_date'])
    df_releases['release_date'] = pd.to_datetime(df_releases['release_date'])
    df_releases['year'] = df_releases['release_date'].dt.year
    df_select_cols = df_releases[['genres', 'title', 'popularity', 'year', 'poster_path',
                                  'overview', 'tagline', 'original_language']].sort_values('year')
    df_select_cols['genres'] = df_select_cols['genres'].apply(transform_categories)
    df_clean = df_select_cols

    return df_clean


def plot_frequency(data: pd.DataFrame) -> pd.Series:
    """
    Plots the total number of films released per year and displays plot in browser.
    :param data: Pandas DataFrame
    :return: Pandas Series containing total number of films released, indexed by year.
    """
    overall_freq = data['year'].value_counts().sort_index()

    plt.style.use('seaborn-whitegrid')

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)
    ax.plot(overall_freq.index, pd.Series(overall_freq.values).rolling(3).mean(), figure=fig)
    plt.title('Films Released by Year')
    plt.ylabel('Number of Films')
    plt.xlabel('Release Year')

    buf = io.BytesIO()
    fig.savefig(buf)
    put_image(buf.getvalue())
    plt.clf()

    return overall_freq


def plot_genre_trends(data: pd.DataFrame, overall_freq: pd.Series, num_genres: int = 8) -> dict:
    """
    Plots number of movies released per year for each genre category as a percentage of total releases that year.
    Displays plot in browser.

    Note that movies are often tagged with multiple genres. This function counts values for unique genres and unique
    genre COMBINATIONS before isolating the basic genres. This usually means that the number of keys in the
    output dictionary will be less than the number passed for num_genres.

    :param data: Pandas DataFrame
    :param overall_freq: Pandas Series containing total number of films released, indexed by year.
    :param num_genres: number of unique genres
    :return: dictionary of genre percents
    """
    top_genres = data['genres'].value_counts().head(num_genres).index
    unique_genres = set(
        ",".join(top_genres)
            .strip()
            .split(",")
    )
    if '' in unique_genres:
        unique_genres.remove('')

    plt.style.use('seaborn-whitegrid')

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)
    plt.title('Historical Trends in Top Genres')
    plt.ylabel('Percent of Films Released')
    plt.xlabel('Release Year')

    genre_percent_dict = {}
    for genre in unique_genres:
        genre_df = data[data['genres'].str.contains(genre)]
        genre_freq = genre_df['year'].value_counts().sort_index()
        genre_percent = (genre_freq / overall_freq * 100).fillna(0)
        genre_percent_dict[genre] = genre_percent
        ax.plot(overall_freq.index, genre_percent.rolling(7).mean(), label=genre)
        ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf)
    put_image(buf.getvalue())
    plt.clf()

    return genre_percent_dict


def map_correlations(genre_percents: dict) -> NoReturn:
    """
    Calculates correlations between genre percentages and displays heatmap in browser.
    :param genre_percents: dictionary of genre percents
    :return:
    """
    genre_list = []
    percent_list = []
    for genre in genre_percents:
        genre_list.append(genre)
        percent_list.append(genre_percents[genre])

    corrs = np.corrcoef(percent_list)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 7)
    sns.heatmap(corrs, linewidths=.5, xticklabels=genre_list, yticklabels=genre_list, square=True)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title('Heatmap Showing Correlations Between Top Genres')

    buf = io.BytesIO()
    fig.savefig(buf)
    put_image(buf.getvalue())
    plt.clf()


def pre_process_text(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes full dataset and returns DataFrame of English only text data with capitalization and punctuation removed.
    :param data: complete dataset
    :return: Pandas DataFrame with preprocessed text data
    """

    df_english = data[data['original_language'] == 'en']
    # select text columns
    text_data = df_english[['title', 'tagline', 'overview']].dropna(subset=['overview']).fillna('')
    # Remove punctuation and quotes
    text_df_preprocessed = text_data.applymap(lambda x: re.sub(r'[,\'".!?]', '', x))
    # Convert to lowercase
    text_df_preprocessed = text_df_preprocessed.applymap(lambda x: x.lower())

    return text_df_preprocessed


def make_wordcloud(preprocessed_text: pd.DataFrame) -> NoReturn:
    """
    Takes preprocessed text DataFrame and displays wordcloud based on movie synopses (from the 'overview' column)
    :param preprocessed_text:
    :return:
    """
    stop_words = stopwords.words('english')
    stop_words_set = set(stop_words)
    overview_long_string = ','.join(list(preprocessed_text['overview'].values))

    wordcloud = WordCloud(width=400, height=200,
                          background_color='white',
                          stopwords=stop_words_set,
                          min_font_size=10).generate(overview_long_string)

    # plot the WordCloud image
    fig = plt.figure(facecolor=None)
    fig.set_size_inches(9, 4.5)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title('Word Cloud Based on Film Synopses')

    buf = io.BytesIO()
    fig.savefig(buf)
    put_image(buf.getvalue())
    plt.clf()


def lemmatize_words(tokenized_data: list, allowed_postags: list = None) -> list:
    """
    Takes tokenized text data and returns lemmas.
    :param tokenized_data:
    :param allowed_postags: optional - specify parts of speach to allow. see Spacy documentation for more info:
    https://spacy.io/api/data-formats
    :return: lemmatized tokens
    """
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    lemmatized_words = []
    for s in tokenized_data:
        doc = nlp(" ".join(s))
        lemmatized_words.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return lemmatized_words


def process_text(preprocessed_text: pd.DataFrame, remove_words: list) -> \
        tuple[list[list[tuple[int, int]]], Dictionary, list]:
    """
    Takes preprocessed text data and processes it for LDA modeling.
    :param preprocessed_text:
    :param remove_words: list of keywords to remove
    :return:
    """

    # remove original keyword(s)
    for keyword in remove_words:
        remove_string = '\w*' + keyword + '\w*'
        preprocessed_text = preprocessed_text.applymap(
            lambda x: re.sub(r'{}'.format(remove_string), '', x))

        preprocessed_text = preprocessed_text.applymap(
            lambda x: re.sub(r'{}'.format(remove_string), '', x))

    # combine text columns
    preprocessed_text['combined'] = preprocessed_text['title'] + ' ' + \
                                    preprocessed_text['tagline'] + ' ' + \
                                    preprocessed_text['overview']

    # tokenize text
    text_data_list = preprocessed_text['combined'].values.tolist()
    data_tokenized = [gensim.utils.simple_preprocess(str(doc), deacc=True) for doc in text_data_list]

    # remove stop words
    stop_words = stopwords.words('english')
    data_tokenized = [[word for word in doc if word not in stop_words] for doc in data_tokenized]

    # identify phrases
    bigram = gensim.models.Phrases(data_tokenized, min_count=5, threshold=10)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_tokenized], threshold=10)
    bigram_model = gensim.models.phrases.Phraser(bigram)
    trigram_model = gensim.models.phrases.Phraser(trigram)
    data_tokenized = [bigram_model[doc] for doc in data_tokenized]
    data_tokenized = [trigram_model[bigram_model[doc]] for doc in data_tokenized]

    # lemmatize words
    data_tokenized = lemmatize_words(data_tokenized)

    # Create Dictionary
    id2word = corpora.Dictionary(data_tokenized)

    # Create Corpus
    # Convert each document into bag-of-words format, i.e. a list of tuples in the form (token_id, token_count)
    corpus = [id2word.doc2bow(doc) for doc in data_tokenized]

    return corpus, id2word, data_tokenized


def lda_modeling(corpus: list, id2word: Dictionary, data_tokenized: list, num_topics: int = 10) -> NoReturn:
    """
    Takes Corpus, Dictionary, tokenized data, and number of topics; builds LDA model and displays
    interactive visualization in browser.
    :param corpus:
    :param id2word:
    :param data_tokenized:
    :param num_topics:
    :return:
    """
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)

    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    LDAvis_prepared = pyLDAvis.prepared_data_to_html(LDAvis_prepared)
    put_html(LDAvis_prepared)

    put_text('Scoring the model...')
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_tokenized, dictionary=id2word,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    put_text('Topic Coherence Score: ', coherence_lda)


def plot_scores(corpus: list, id2word: Dictionary, data_tokenized: list, max_k: int) -> NoReturn:
    """
    Takes Corpus, Dictionary, tokenized data, and maximum number of topics (specified by user). Builds and scores an
    LDA model for each k between 1 and the maximum specified. Displays plot of the coherence scores.
    :param corpus:
    :param id2word:
    :param data_tokenized:
    :param max_k:
    :return:
    """
    coherence_scores = []
    for k in range(1, max_k + 1):
        num_topics = k
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics)
        coherence_model_lda = CoherenceModel(model=lda_model,
                                             texts=data_tokenized,
                                             dictionary=id2word,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_scores.append(coherence_lda)
        put_text('k =', k, '--->', coherence_lda)

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)
    ax.plot(range(1, max_k + 1), coherence_scores)
    plt.xlabel("Number of Topics (k)")
    plt.ylabel("Coherence Score")
    plt.xticks(range(1, max_k + 1, 2))

    buf = io.BytesIO()
    fig.savefig(buf)
    put_image(buf.getvalue())
    plt.clf()


def task_func() -> NoReturn:
    """
    Film Data Explorer
    """
    api_key = input("Enter TMDB API key：", type=TEXT, required=True)
    keywords = input("Enter keywords: ", type=TEXT, required=True)
    keyword_list = keywords.split(',')

    ids = tmdb_keyword_query(keywords, api_key)

    proceed = select(label="Proceed?",
                     options=['Yes', 'No'])

    if proceed == 'No':
        put_text('Refresh browser to start new query.')
        return

    else:
        put_text('Building dataset...')
        data_df = build_dataset(ids, api_key)

    process = select(label='How would you like to explore the data?',
                     options=['Automate in browser', 'Download'])

    if process == 'Automate in browser':
        put_text('Cleaning data and preparing visualizations...')

        # clean data
        clean_df = clean_data(data_df)

        # visualize data
        frequency = plot_frequency(clean_df)
        genre_percent_dict = plot_genre_trends(clean_df, frequency)
        map_correlations(genre_percent_dict)

        # isolate text data and preprocess it
        text_df_preprocessed = pre_process_text(clean_df)

        # create and display wordcloud
        make_wordcloud(text_df_preprocessed)

        # topic modeling

        build_model = select(label='Would you like to build a topic model based on text data?',
                             options=['Yes', 'No'])

        if build_model == 'Yes':
            put_text('Building and visualizing Latent Dirichlet allocation (LDA) model...')

            # process text and generate topic model
            corpus, id2word, data_tokenized = process_text(text_df_preprocessed, keyword_list)
            lda_modeling(corpus, id2word, data_tokenized)

            find_optimal = select(label='Would you like to search for a more optimal number of topics? This takes a '
                                        'while!',
                                  options=['Yes', 'No'])

            if find_optimal == 'Yes':
                max_k = select(label='Select maximum number of topics:',
                               options=[int(i) for i in [1, 2, 3]])

                put_text('Building and scoring models with number of topics (k) between 1 and', max_k, '...')

                plot_scores(corpus, id2word, data_tokenized, max_k)

                new_model = select(label='Would you like to tune k and build a new model?',
                                   options=['Yes', 'No'])

                if new_model == 'Yes':
                    num_topics = select(label='How many topics?',
                                        options=[int(i) for i in range(1, 31)])
                    num_topics = int(num_topics)

                    put_text('Building and visualizing Latent Dirichlet allocation (LDA) model with',
                             num_topics, 'topics...')
                    # TODO fix alignment

                    lda_modeling(corpus, id2word, data_tokenized, num_topics=num_topics)


app.add_url_rule('/', 'webio_view', webio_view(task_func),
                 methods=['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
    app.run(debug=True)
