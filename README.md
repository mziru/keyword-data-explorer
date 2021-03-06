# movie-keyword-data-explorer
A web application for wrangling, exploring, and modeling movie data based on keywords. 

The app uses the open-source TMDB web API to generate a custom dataset based on user input keyword(s), automatically cleans the data, outputs some exploratory visualizations, and gives the user options to train, visualize, and tune a Latent Dirichlet allocation (LDA) topic model based on natural language data (titles, taglines, and synopses).

Technologies:
- Natural Language Processing and Modeling:
   -	Gensim
   -	spaCy 
   -	NLTK
- Visualizations:
    -	pyLDAvis
    -	Matplotlib
    -	seaborn
    -	wordcloud
- Math and Data Structures:
    -	Pandas
    -	NumPy
- Browser Interface:
    -	PyWebIO
    -	Flask

Note: this is a work in progress with basic functionality and a rudimentary UI for now.

Below is some sample output based on a query using the keyword 'alien':

The keyword 'alien' returned a dataset of 1,057 movies. The user has the option to download the data as a .csv file or automate the exploration process in the browser.  

<img src="https://github.com/mziru/keyword-data-explorer/blob/master/readme%20images/success.png?raw=true" width="600">

If the user chooses to automate the process, the first plot shows the total number of films released per year that match the keyword.

<img src="https://github.com/mziru/keyword-data-explorer/blob/master/readme%20images/release_frequency.png?raw=true" width="600">

The next plot shows historical trends in the most prevalent genres as a percent of the total number of films released each year. (Some films are tagged with multiple genres, so the sum of percents across genres may be greater than 100.)

<img src="https://github.com/mziru/keyword-data-explorer/blob/master/readme%20images/genre_plot.png?raw=true" width="600">

Next, a heatmap showing correlations between broad genre categories. (It makes some intuitive sense, here, that Science Fiction releases would be negatively correlated with Documentary releases—but what exactly happened with alien movies in the late 1970s?...)

<img src="https://github.com/mziru/keyword-data-explorer/blob/master/readme%20images/heatmap.png?raw=true" width="600">

Before training a model, the app does some initial language processing and displays a word cloud, which shows the relative frequencies of words across all of the movie synopses (and validates, in this case, that 'alien' appears most frequently.)

<img src="https://github.com/mziru/keyword-data-explorer/blob/master/readme%20images/word_cloud.png?raw=true" width="600">

The user now has the option to train a topic model in order to try to uncover "latent" semantic structures in the language data that might be of more interpretive value than the basic genre categories. Latent Dirichlet allocation is an unsupervised machine learning technique; the only parameter that must be specified is the number of topics (k). The first model uses k=10 by default. The user will be able to specify other values for k after the default model is trained and scored.
Before training, the language data goes through a few more processing steps to
-	remove the original keywords,
-	combine the text of titles, taglines, and synopses into a single document for each movie in the dataset,
-	identify phrases (bigrams and trigrams),
-	lemmatize words,
-	and build a dictionary and corpus to use for modeling.

Once the model-training is complete, the app leverages the PyLDAvis library to create an interactive visualization within the browser to help the user interpret the topics.

<img src="https://github.com/mziru/keyword-data-explorer/blob/master/readme%20images/lda_viz.png?raw=true" width="600">

The user then has the option to try to find a more optimal number of topics by iterating through values for k (up to a maximum specified by the user), training a separate model for each k, and plotting a topic coherence score for each model. (Note that, in this example, the CV score was used to measure coherence becasue it is Gensim's default metric, but other metrics are available.) 

<img src="https://github.com/mziru/keyword-data-explorer/blob/master/readme%20images/coherence_plot.png?raw=true" width="600">

The user can now train and visualize a new model with a custom k. In this case, the "elbow" heuristic suggests that 7 would be a more optimal number of topics than the default of 10.

That's all for now! Some ideas for future improvements: more interactive visualizations, more hyperparameter tuning within the browser interface, options to download/export processed data and models, some different ways to visualize and explore the topic groupings--e.g. to see more clearly how the latent semantic structures revealed by the model can be interpreted.  





