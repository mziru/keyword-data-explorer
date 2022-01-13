import ast
import io
import json
import math
import os
import re

import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
import seaborn as sns
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from pywebio.input import select, input, TEXT
from pywebio.output import put_text, put_image, put_html
from pywebio.platform.flask import webio_view
from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
from wordcloud import WordCloud

from flask import Flask

app = Flask(__name__)


# TMDB developer keys are available from TMDB: https://developers.themoviedb.org/3/getting-started/introduction

def tmdb_keyword_query(kw_list, api_key):
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


def build_dataset(id_list, api_key):
    """
    Builds dataset using TMDB web API
    :param id_list: list of TMDB unique ids
    :param api_key:
    :return: Pandas Dataframe
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


def fix_list(value_list):
    string = str(value_list)
    dicts_list = ast.literal_eval(string)
    string_list = []
    for d in dicts_list:
        string_list.append(d['name'])
    new_string = ','.join(string_list)
    return new_string


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


# ids = tmdb_keyword_query(['bike', 'bicycle', 'bmx'], '1a639c8e33a30017bb883bf46d80f183')
def task_func():
    api_key = input("Enter TMDB API key：", type=TEXT, required=True)
    keywords = input("Enter keywords: ", type=TEXT, required=True)

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

        df = data_df.dropna(subset=['release_date'])
        df['release_date'] = pd.to_datetime(df['release_date'])
        df['year'] = df['release_date'].dt.year
        df = df[['genres', 'title', 'popularity', 'year', 'poster_path',
                 'overview', 'tagline', 'original_language']].sort_values('year')
        df['genres'] = df['genres'].apply(fix_list)

        plt.style.use('seaborn-whitegrid')

        overall_freq = df['year'].value_counts().sort_index()
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

        top_genres = df['genres'].value_counts().head(8).index
        unique_genres = set(
            ",".join(top_genres)
                .strip()
                .split(",")
        )
        if '' in unique_genres:
            unique_genres.remove('')

        fig, ax = plt.subplots()
        fig.set_size_inches(9, 4.5)
        plt.title('Historical Trends in Top Genres')
        plt.ylabel('Percent of Films Released')
        plt.xlabel('Release Year')

        genre_percent_dict = {}
        for genre in unique_genres:
            genre_df = df[df['genres'].str.contains(genre)]
            genre_freq = genre_df['year'].value_counts().sort_index()
            genre_percent = (genre_freq / overall_freq * 100).fillna(0)
            genre_percent_dict[genre] = genre_percent
            ax.plot(overall_freq.index, genre_percent.rolling(7).mean(), label=genre)
            ax.legend()

        buf = io.BytesIO()
        fig.savefig(buf)
        put_image(buf.getvalue())
        plt.clf()

        genre_list = []
        percent_list = []
        for genre in genre_percent_dict:
            genre_list.append(genre)
            percent_list.append(genre_percent_dict[genre])

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

        df_english = df[df['original_language'] == 'en']
        text_data = df_english[['title', 'tagline', 'overview']].dropna()

        # Remove punctuation and quotes
        text_data_processed = text_data.applymap(lambda x: re.sub(r'[,\'".!?]', '', x))
        # Convert the titles to lowercase
        text_data_processed = text_data_processed.applymap(lambda x: x.lower())

        stop_words = stopwords.words('english')
        stop_words_set = set(stop_words)
        overview_long_string = ','.join(list(text_data_processed['overview'].values))

        wordcloud = WordCloud(width=400, height=200,
                              background_color='white',
                              stopwords=stop_words_set,
                              min_font_size=10).generate(overview_long_string)

        # plot the WordCloud image
        fig = plt.figure(facecolor=None)
        fig.set_size_inches(9, 4.5)
        plt.imshow(wordcloud)
        plt.axis("off")
        # plt.tight_layout(pad=0)
        plt.title('Word Cloud Based on Film Synopses')

        buf = io.BytesIO()
        fig.savefig(buf)
        put_image(buf.getvalue())
        plt.clf()

        # topic modeling
        build_model = select(label='Would you like to build a topic model based on text data?',
                             options=['Yes', 'No'])

        if build_model == 'Yes':
            remove_keywords = select(label='Exclude original keyword(s) from topics?',
                                     options=['Yes', 'No'])
            put_text('Exclude original keyword(s):', remove_keywords)

            title_tagline = select(label='Incorporate movie titles and taglines when modeling topics?',
                                   options=['Yes', 'No'])
            put_text('Incorporate titles and taglines:', title_tagline)

            lemmatize_words = select(label='Lemmatize words?',
                                     options=['Yes', 'No'])
            put_text('Lemmatize words:', lemmatize_words)

            num_topics = input('How many topics? (Default = 10)', type=TEXT, required=False)
            num_topics = int(num_topics)
            put_text('Number of topics:', num_topics)
            put_text('Building and visualizing Latent Dirichlet allocation (LDA) model...')
            # text_data = df[['title', 'tagline', 'overview']].dropna()

            # remove original keywords

            # remove_string = '\w*{}\w*'.format(kw_list)
            #
            # text_data['overview_processed'] = text_data['overview'].map(
            #     lambda x: re.sub(r'{}'.format('\w*robot\w*'), '', x))

            text_data_list = text_data_processed['overview'].values.tolist()
            data_words = list(sent_to_words(text_data_list))

            # remove stop words
            data_words = remove_stopwords(data_words, stop_words)
            # put_text(data_words[:1][0][:30])

            # Create Dictionary
            id2word = corpora.Dictionary(data_words)
            # Create Corpus
            texts = data_words
            # Term Frequency
            corpus = [id2word.doc2bow(text) for text in texts]

            # number of topics
            # num_topics = 2
            # Build LDA model
            lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=num_topics)
            # Print the Keyword in the 10 topics
            # put_text(lda_model.print_topics())
            doc_lda = lda_model[corpus]

            # Visualize the topics

            LDAvis_data_filepath = os.path.join('./ldavis_prepared_' + str(num_topics))
            LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
            LDAvis_prepared = pyLDAvis.prepared_data_to_html(LDAvis_prepared)
            put_html(LDAvis_prepared, scope='ROOT').style('margin-left: 0px')


app.add_url_rule('/', 'webio_view', webio_view(task_func),
                 methods=['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
    app.run(debug=True)
