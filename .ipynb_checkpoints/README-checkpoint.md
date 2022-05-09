# Hybrid-Movie-Recommender   
#### Ankita Patil
---
## Executive Summary

Recommendation systems are a way of filtering and recommending information based on a user's likes or preferences. They are used by various platforms like streaming services, music services, e-commerce and social media platforms to increase user engagement by solving the problem of information overload, filtering and recommending the right content to the right customer.

Using user data from MovieLens and movie data from IMDb and TMDB, I have built a hybrid movie recommender system that combines content-based and user-based collaborative filtering. The content-based recommender uses TF-IDF, Sentiment Analysis and Cosine Similarity to find relevance in the movies and the user-based collaborative recommender uses Cosine Similarity to find relevance in the user ratings. The hybrid recommender overcomes the flaws namely cold start, inadequate diversity and subject-matter expertise dependency and combines the strengths of individual filtering techniques.

The hybrid movie recommender has been used to build an app where users can rate the movies they watched and get personalized movie recommendations. Although the model benefits from a user entering a high number of movies and ratings, it also gives successful recommendations to a user entering low number of movies and ratings based on the feedback received from the users.


### Problem Statement
Given a database of users and their movie ratings, can we build a recommender that predicts a ranked list of movie recommendations that the user would like to watch?



### Data Sources

- **[MovieLens](https://grouplens.org/datasets/movielens/latest/)**: The user rating data is taken from MovieLens. MovieLens is a movie recommendation service maintained by GroupLens, a research group at the University of Minnesota and provides data on movies and user ratings. The dataset used for this project is the `MovieLens Latest Dataset - Small`. The small dataset, which consists of 100,000 ratings on 9724 movies by 610 users, was chosen due to limitations in available computational power as compared to the `MovieLens Latest Dataset - Full`, which contains 27,000,000 ratings. The small dataset was curated from the full dataset by keeping users that have rated minimum 20 movies. This is a more effective method rather than randomly choosing users from the full dataset and running into an issue of data not missing at random.

- **[IMDb](https://www.imdb.com/interfaces/)**: Movie popularity data is taken from IMDb. Average rating and number of votes for each movie is extracted from this dataset. The IMDb database is huge and much more popular amongst users than TMDB.

- **[TMDB](https://www.themoviedb.org/?language=en-US)**: Movie popularity, tagline, synopsis, keywords, director and cast data comes from TMDB. The TMDB database is less popular than IMDb, but unlike IMDb, it is driven by the user community and has much more diverse features as mentioned above. This data is taken from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

### Data Exploration 

![](https://github.com/ankitadpatil/Hybrid-Movie-Recommender/blob/main/images/ratings.png)
![](https://github.com/ankitadpatil/Hybrid-Movie-Recommender/blob/main/images/average_rating.png)
![](https://github.com/ankitadpatil/Hybrid-Movie-Recommender/blob/main/images/genre_popularity.png)
![](https://github.com/ankitadpatil/Hybrid-Movie-Recommender/blob/main/images/genre_ratings.png)
![](https://github.com/ankitadpatil/Hybrid-Movie-Recommender/blob/main/images/ratings_per_movie.png)

### Approach

The project is built on the assumption that users have only watched the movies they have rated. If a user has not rated a movie, then the user has not seen the movie and can be recommended to the user.

Recommender systems are used to predict a user's rating on an unseen item and recommend the item if the predicted score is high. Information filtering can be achieved by the following four ways:   

1. **Popularity-Based Recommender**:
- A popularity-based recommender filters information based on popularity of the item and make recommendations irrespective of the user preference.
- The most evident disadvantage of this recommender is that it gives out the same predictions to all the users without any personalization. Popular movies are already known to users and do not require being recommended.
2. **Content-Based Recommender**:
- A content-based recommender filters information based on a user and his/her interactions only. It takes into consideration features of interacted items and the user feedback, and makes recommendations similar to these items.
- This filter is implemented by considering features like movie synopsis, tagline, keywords, director etc. and whether the user liked a movie or not. Based on the movie features mentioned above and their including sentiment analysis, a movie-movie (item-item) similarity matrix is created using TF-IDF, where the recommender looks for similar new items, ranks them by similarity and recommends them by decreasing order of similarity.
- As content-based recommender works solely on a user's interaction with items, it does not require data from other users, making it easy to scale. This is advantageous when businesses are starting out new and do not have a lot of user interaction data. This recommender is also better at giving out recommendations to new users as compared to collaborative filtering when you do not have a lot of data on the users.
- It is also important to note that this recommendation can only get as good as it's features. And since these are engineered manually, this model highly relies on domain knowledge for its performance. Additionally, the model's recommendations are less diverse in recommending new items as it relies on user's existing interests. It can also recommend items with low ratings.

**Cosine Similarity** - Cosine similarity is a metric to measure similarity between two vectors. Smaller the angle between the two vectors, larger the similarity between the two. Similarity between two movies or two users can be quantified using cosine similarity. If we consider movies and users as vectors where features form dimensions of the vectors, the cosine of angle between these two vectors (movies/users) gives similarity between the two movies or two users. The mathematical formula for calculating cosine similarity is -

![](https://github.com/ankitadpatil/Hybrid-Movie-Recommender/blob/main/images/CosineSimilarity.png)

**TF-IDF (Term Frequency - Inverse Document Frequency)** is a measure of how relevant a word is to a document in a collection of documents. TF-IDF caluculates frequency of a word in a document and in a collection of documents. The rank of every word is directly proportional to its frequency in the document, but inversely proportional to its frequency across multiple documents. So if a word is very common and appears in the context of many movies, it is ranked lower and if the word appears in the context of only one movie, it is ranked higher. TF-IDF is primarily used to convert text into numbers. 

3. **Collaborative Recommender**:
- A memory-based collaborative recommender filters information based on other users' interactions. The core idea behind collaborative filtering is that people (users) that share interests (similar ratings) on certain things (items) are more likely to share interests on other things (other items). There are two types of collaborative filtering:
  * **User-Based Approach** - It uses ratings given by all the users (user behavior) to find similarity between the users. It then finds users similar to the target user, sorts these users by similarity scores. It then makes recommendations based on these user' interactions. You can see user-based collaborative filtering at work when you come across `because you watched this we recommend you something (a list)`.
  * **Item-based Approach** - It uses ratings given by all the users (user behavior) to find similarity between the items instead of users. It then finds items similar to the items watched by the target user and recommends items by similarity score. You can see item-based collaborative filtering approach being implemented when you come across `people who liked this also liked something (a list)`.
- Both approaches work on user-item interactions matrix and have an advantage based on the scenarios in which they are used. User-based approach is implemented in the project as number of users is less than number of movies (items). This saves computation time and power required to create a user-user interaction matrix. Item-based approach is used when number of items are less than number of users, and less computation time and power is available to create an item-item interaction matrix.
- A user-based collaborative filtering recommender is implemented in this project by creating a user-movie (user-item) interaction matrix based on user ratings, and identifying similar users based on the interactions. A target user's rating on an unseen movie is predicted based on ratings given by similar users identified by the interaction matrix, on the movie in consideration. The predicted rating is calculated by taking a weighted average of similar users' ratings with weights being similarity score with the target user. Since the predicted rating is a compounded result of ratings given by similar users, it is necessary to set a positive threshold of similarity score, below which similar users are not considered in the prediction of rating. This is required to avoid bringing down of the rating prediction due to user input from less similar users.
- As user-based collaborative recommender depends on user-user interactions, it has a greater diversity than content-based recommender. Its performance is not based on domain knowledge. and is benefits from a growing dataset. It is also worth mentioning that it cannot function effectively in case of sparse data in user-item matrix. This will remain as an ongoing problem as not all users rate their interaction with items.
- As mentioned earlier, collaborative filtering suffers when you have little information on new users and less user interactions. This is also called as 'cold start problem'. Additionally, user-user interaction computation needs to be performed more frequently as new users get added. This poses a problem for space requirement data storage, hardware requirement and data maintenance.

4. **Hybrid Recommender**:
- As mentioned above, both content-based and user-based collaborative filtering models have their own pros and cons. Hybrid recommender system ensembles the two approaches with an aim to combine their strengths and overcome their weaknesses. This can be achieved by making content-based and user-based collaborative predictions separately and then combining the two. Content-based predictions solve the problem of cold-start and sparsity of user-item interaction data and user-based collaborative predictions solve the problem of inadequate diversity, low rating recommendations and subject-matter expertise dependency.
- The recommender is built by combining the two methods and taking the average of the cosine similarity scores to create a ranked list of predictions.

### Workflow
1. Data Collection and Wrangling
2. Exploratory Data Analysis
3. Feature Engineering
4. Natural Language Processing
5. Building Hybrid Recommendation Engines
6. Building Recommender App


### Data Dictionary

|Feature|File|Type|Description|
| --- | --- | --- | --- |
| **movie_id** | content | *object* | Unique movie id  given by MovieLens
| **title** | content | *object* | Movie title with year of release
| **genres** | content | *object* | Movie genres
| **year** | content | *object* | Year of release
| **tmdb_id** | content | *object* | Unique movie id given by TMDB (The Movie Database)
| **imdb_id** | content | *object* | Unique movie id given by IMDb (Internet Movie Database)
| **overview** | content | *object* | Synopsis of the movie  
| **tagline** | content | *object* | Slogan/catchphrase for the movie
| **tmdb_rating** | content | *float*  | Average rating of a movie on TMDB
| **tmdb_votes** | content | *int* |  Number of votes/ratings received on the movie on TMDB
| **imdbId_rating** | content | *float* | Average rating of a movie on IMDb
| **imdbId_votes** | content | *int* |  Number of votes/ratings received on the movie on IMDb
| **keywords** | content | *object* |  Movie plot keywords
| **director** | content | *object* |  Director of the movie
| **cast** | content | *object* |  Cast of the movie (up to 4)
| **sentiment_score** | content | *float* |  Polarity score on the movie's overview, tagline and keywords
| **user_id** | ratings_title | *int* | Unique user id from MovieLens database (610 unique)
| **movie_id** | ratings_title | *int* | Unique movie id  from the MovieLens database
| **rating** | ratings_title | *float* | Movie rating by the user: 0.5 - 5 and 0 (`not rated`)
| **title** | ratings_title | *object* | Movie title  with year of release
| **genres** | ratings_title | *object* | Movie genres


### Conclusion
Three types of recommender systems were built and combined for this movie recommender system.
1. Content-based recommender generated ranked list of movie recommendations based on content similarity in user's viewing history. Movie features like genre, plot synopsis, tagline, keywords, director, cast and sentiment analysis were used to find similarity in content and cosine similarity was used to quantify this similarity. These  recommendations however lacked diversity and ability to assess whether these recommendations were popular.
2. User-based (Memory-based) collaborative recommender generated ranked list of movie recommendations based on ratings given by similar users. User-based approach was chosen because the dataset contains more items and less users, and user-user interaction matrix will take less computational time than item-item interaction matrix. This recommender is highly dependent on historical data and suffers from the problem of cold start and data sparsity.
3. Hybrid recommender, which was built combining the 2 recommender systems above, was able to give more accurate movie recommendations than individual recommender systems. This system was able to partially address weaknesses of individual recommenders. It addressed the problem of cold-start and data sparsity in collaborative recommender through content-based filtering and lack of diversity, low rating recommendations and subject-matter expertise dependency in content-based recommender through collaborative filtering. The problem of speed and scalability still persists as the collaborative filter uses memory-based algorithm that requires storing all the user-item interactions in order to make recommendations.
4. Model-based algorithms can be implemented to solve the problem of speed and scalability because unlike memory-based systems, they use partial data. Since these algorithms use partial data instead of complete data to make predictions, it leaves recommender model susceptible to inaccurate predictions. These algorithms also suffer from the problem of data sparsity in user-item interaction matrix and the predictions are not as easy to interpret as memory-based algorithms.
5. Metrics that are often used to evaluate models are not effective in evaluating recommendation systems. Metrics like RMSE (Root Mean Square Error) and Hit Rate can be indicators while developing recommender systems offline. But these cannot give real indication of success unless the system has been tried on real users. User reaction is the real test of a recommendation system.
6. The hybrid recommender system built was deployed on Streamlit and tried by a number of users, with mostly positive responses. The model can be further improved by using A/B testing in the app, which will help in improving recommendations by making controlled experiments and collecting data on the results.


### Limitations

The project is based on the assumption that users have only watched movies that they have rated. But in the real world, not all users who watch movies provide ratings. In such cases, companies rely on information like hours viewed, with the occasional 'Still watching' prompts. Hours viewed can be a strong indicator of whether a user liked a movie. As user ratings is the only data available, it has been used as the only metric of interaction.

The recommendation system uses memory-based algorithm that relies heavily on system computational power. This approach results into a slow and difficult to scale systems. Model-based algorithms can solve these problems to a large extent. Model-based collaborative filtering is not part in current scope of this project at this point. Eventually, I intend to incorporate matrix factorization based algorithms Singular Value Decomposition (SVD) and Singular Value Decomposition ++ (SVD++) to build a model-based hybrid recommendation engine.


### Tools and Packages

- **Stack**: Python, Git, Markdown
- **Web Scraping**: requests, Beautifulsoup, json, pickle
- **Feature Engineering**: Natural Language Processing, TF-IDF, regex
- **Modeling/Machine Learning**: scikit-learn, scipy, numpy, pandas
- **Data Visualization**: Matplotlib, Seaborn
- **Web App**: Streamlit
