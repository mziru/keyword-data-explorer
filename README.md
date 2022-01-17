# movie-keyword-data-explorer
A web application for wrangling, exploring, and modeling movie data based on keywords. 

The app leverages the open-source Gensim API for unsupervised semantic modeling. It uses the open-source TMDB web API to generate a custom dataset based on user input keyword(s), automatically cleans the data, outputs some exploratory visualizations, and gives the user options to train, visualize, and tune an LDA topic model based on natural language data (titles, taglines, and synopses).

Note: this is a work in progress with basic functionality and a rudimentary UI for now. Some ideas for improvement: more interactive visualizations, more hyperparameter tuning within the browser UI, options to download/export data and models,  

Here's some sample output based on a query using the keyword 'alien':

<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/release_frequency.png?raw=true" width="600">
<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/genre_plot.png?raw=true" width="600">
<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/heatmap.png?raw=true" width="600">
<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/word_cloud.png?raw=true" width="600">
<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/lda_viz.png?raw=true" width="600">
<img src="https://github.com/mziru/movie-keyword-data-explorer/blob/master/readme%20images/coherence_plot.png?raw=true" width="600">





