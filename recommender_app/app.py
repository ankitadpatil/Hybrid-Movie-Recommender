from hashlib import new
from ssl import Options
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from functions import *

st.header('Personalized Movie Recommendations')

#Data imports
df_content = pd.read_csv('clean_content.csv')
df_user = pd.read_csv('ratings_title.csv')
df_user.rename(columns={'userId':'user_id', 'movieId':'movie_id'}, inplace=True)
content_similarity = pickle.load(open('movie_similarity_matrix.pkl', 'rb'))
df_content_sim = pd.DataFrame(content_similarity, index = df_content['title'].values, columns= df_content['title'].values)


#Get data from the user
new_user_data = []
number = int(st.number_input('How many movies would you like to rate?', min_value=3,value=3,step=1))
#Varible to assign unique key for every iteration in the for loop
current_line_number = 0
options = df_content['title'].values.tolist()
for _ in range(number):
    movie = st.selectbox('Movie title',key=str(current_line_number), options=options)
    rating = st.select_slider('Rate the movie',options=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], key=str(current_line_number))
    new_user_data.append((movie,rating))
    current_line_number += 1

if st.button('Get Recommendations'):

    #Add new_user_data to user database
    new_userId = df_user['user_id'].sort_values().values[-1] + 1
    new_user = []
    for movie,rating in new_user_data:
        new_ratings = {}
        new_ratings['user_id'] = new_userId
        new_ratings['rating'] = rating
        new_ratings['movie_id'] = df_content.loc[df_content['title'] == movie, 'movie_id'].values[0]
        new_ratings['title'] = movie
        new_ratings['genres'] = df_content.loc[df_content['title'] == movie, 'genres'].values[0]
        new_ratings['year'] = df_content[df_content['title'] == movie]['year'].values[0]
        new_user.append(new_ratings)

    df_new_user = pd.DataFrame(new_user).drop_duplicates()

    #Add the new user to the df_user dataframe
    df_user = pd.concat([df_user, df_new_user])

    #Create User-Item Matrix
    user_item = df_user.pivot_table(values = 'rating', index = 'user_id', columns= 'title')

    #Normalize User-Item matrix
    norm_user_item = user_item.subtract(user_item.mean(axis=1), axis = 'rows')

    #User-User similarity matrix
    user_similarity = cosine_similarity(sparse.csr_matrix(norm_user_item.fillna(0)))
    df_user_sim = pd.DataFrame(user_similarity, index=user_item.index, columns=user_item.index)

    #st.table(hybrid_recommender(new_userId))

    def get_content_similar_movies(user):
    
        #Current/target user
        df_current_user = df_user[df_user['user_id'] == user]
    
        #Movies watched by the current/target user
        user_watched_movies = df_current_user['title'].values
    
        #User's mean rating
        user_mean_rating = df_current_user['rating'].mean()
    
        #Filter the list of movies by like/dislike based on user's rating
        user_movies = []
        for movie in user_watched_movies:
            if df_current_user[df_current_user['title'] == movie]['rating'].values >= user_mean_rating:
                user_movies.append(movie)
            
        #Create an empty dataframe to store movie recommendations for each movie seen by the user
        similar_movies = pd.DataFrame()
        #Loop through each movie seen by the user
        for movie in user_movies:
            #Add similarity score for each movie with user_movie
            #Remove movies that the user has already seen
            similar_movies = similar_movies.append(df_content_sim[movie].drop(user_watched_movies))
        #Add the similarity score of each movie and select the movies with high scores
        content_rec = pd.DataFrame(similar_movies.sum()).reset_index().rename(columns={'index': 'title',
                        0: 'content_similarity'})
        return pd.merge(df_content[['title', 'genres']], content_rec, how='inner').sort_values(by='content_similarity', ascending=False)
    def get_user_similar_movies(user, similarity_threshold):
    
        #Extract similar users and their similarity score with the target user
        similar_users = df_user_sim[df_user_sim[user] > similarity_threshold][user].sort_values(ascending=False)[1:]
    
        #Extract movies watched by the target user and their score with the target user
        target_user_movies = norm_user_item[norm_user_item == user].dropna(axis =1, how= 'all')
    
        #Extract movies watched by similar users and their score with the similar users
        similar_user_movies = norm_user_item[norm_user_item.index.isin(similar_users.index)].dropna(axis=1, how = 'all')
    
        #Keep the movies watched by similar users but not by the target user: 
        for column in target_user_movies.columns: 
            if column in similar_user_movies.columns:
                similar_user_movies.drop(column, axis=1, inplace=True)
            
        #Weighted average
        movie_score = {}
        #Loop through the movies seen by similar users
        for movie in similar_user_movies.columns:
            #Extract the rating for each movie
            movie_rating = similar_user_movies[movie]
            #Variable to calculate numerator of the weighted average
            #This must be calculated for each movie
            numerator = 0
            #Variable to calculate the denominator of the weighted average
            denominator = 0
            #Loop through the similar users for that movie
            for user in similar_users.index:
                #If the similar user has seen the movie
                if pd.notnull(movie_rating[user]):
                    #Weighted score is the product of user similarity score and movie rating by the similar user
                    weighted_score = similar_users[user] * movie_rating[user]
                    numerator += weighted_score
                    denominator += similar_users[user]
            movie_score[movie] = numerator / denominator
        #Save the movie and the similarity score in a dataframe
        movie_score = pd.DataFrame(movie_score.items(), columns=['title', 'user_similarity'])
        user_rec = pd.merge(df_content[['title','genres','year']], movie_score[['title', 'user_similarity']], how='inner')
        return user_rec.sort_values(by=['user_similarity', 'year'], ascending=False)
        

    def hybrid_recommender(user):
        content_user_scores = pd.merge(get_content_similar_movies(user), get_user_similar_movies(user, 0.1))
        content_user_scores['similarity_score'] = (content_user_scores['content_similarity'] + content_user_scores['user_similarity']) / 2
        top_scores = content_user_scores.sort_values(by='similarity_score', ascending=False)[:10]
        #recommendations = pd.merge(df_content[['title','vote_average', 'vote_count']], top_scores[['title', 'similarity_score']], on='title')
        #recommendations.rename(columns={'title': 'Movie Title', 'vote_average' : 'TMDb Rating', 'similarity_score' : 'Similarity Score'}, inplace=True)
        #return recommendations.sort_values(by='Similarity Score', ascending=False)
        recommendations = pd.merge(df_content[['title','genres','imdb_rating', 'tmdb_rating']], top_scores[['title','similarity_score']], on='title')
        recommendations.rename(columns={'title':'Movie Title', 'imdb_rating': 'IMDb Rating', 'tmdb_rating':'TMDB rating', 'similarity_score':'Similarity Score'}, inplace=True)
        return recommendations.sort_values(by='Similarity Score', ascending=False)

    st.table(hybrid_recommender(new_userId))