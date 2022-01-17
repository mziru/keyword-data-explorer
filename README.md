# movie-keyword-data-explorer
A web application for wrangling, exploring, and modeling movie data based on keywords. 

The app leverages the open-source Gensim API for unsupervised semantic modeling. It uses the open-source TMDB web API to generate a custom dataset based on user input keyword(s), automatically cleans the data, outputs some exploratory visualizations, and gives the user options to train, visualize, and tune a Latent Dirichlet allocation (LDA) topic model based on natural language data (titles, taglines, and synopses).

Note: this is a work in progress with basic functionality and a rudimentary PyWebIO UI for now. Some ideas for improvement: more interactive visualizations, more hyperparameter tuning within the browser interface, options to download/export processed data and models, some different ways to visualize and explore the topic groupings--e.g. to see more clearly how the "latent" semantic structures revealed by the model can be interpreted.  

Below is some sample output based on a query using the keyword 'alien':

The keyword 'alien' returned a dataset of 1,057 movies. The user has the option to download the data as a .csv file or automate the exploration process in the browser.  

<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/success.png?raw=true" width="600">

If the user chooses to automate the process, the first plot shows the total number of films released per year that match the keyword.

<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/release_frequency.png?raw=true" width="600">

The next plot shows historical trends in the most prevalent genres as a percent of the total number of films released each year. (Some films are tagged with multiple genres, so the sum of percents across genres may be greater than 100.)

<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/genre_plot.png?raw=true" width="600">

Next, a heatmap showing correlations between broad genre categories. (It makes some intuitive sense, here, that Science Fiction releases would be negatively correlated with Documentary releases—but what exactly happened with alien movies in the late 1970s?...)

<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/heatmap.png?raw=true" width="600">

Before training a model, the app does some light language processing and displays a word cloud, which shows the relative frequencies of words across all of the movie synopses (and validates, in this case, that 'alien' appears most frequently.)

<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/word_cloud.png?raw=true" width="600">
<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/lda_viz.png?raw=true" width="600">
<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/coherence_plot.png?raw=true" width="600">





