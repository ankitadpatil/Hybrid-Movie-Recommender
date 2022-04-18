import numpy as np
import pandas as pd


def get_content_similar_movies(user):
    
    #Extract the movies watched by the user
    user_movies = df_user[df_user['userId'] == user]['title'].values
    
    #Create an empty dataframe to store movie recommendations for each movie seen by the user
    similar_movies = pd.DataFrame()
    
    #Loop through each movie seen by the user
    for movie in user_movies:
        #Add similarity score for each movie with user_movie
        #Remove movies that the user has already seen
        similar_movies = similar_movies.append(df_content_sim[movie].drop(user_movies))
    #Add the similarity score of each movie and select the movies with high scores
    return pd.DataFrame(similar_movies.sum()).reset_index().rename(columns={'index': 'title',
                        0: 'content_similarity'}).sort_values(by='content_similarity', ascending=False)


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
    return movie_score.sort_values(by='user_similarity', ascending=False)
    

def hybrid_recommender(user):
    content_user_scores = pd.merge(get_content_similar_movies(user), get_user_similar_movies(user, 0.1))
    content_user_scores['similarity_score'] = (content_user_scores['content_similarity'] + content_user_scores['user_similarity']) / 2
    top_scores = content_user_scores.sort_values(by='similarity_score', ascending=False)[:10]
    recommendations = pd.merge(df_content[['title','vote_average', 'vote_count']], top_scores[['title', 'similarity_score']], on='title')
    return recommendations.sort_values(by='similarity_score', ascending=False)