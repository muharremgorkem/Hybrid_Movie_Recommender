
########################################################################################
# # Hybrid Movie Recommendation System: Personalized Suggestions using MovieLens Dataset
########################################################################################

######################
# 1. Business Problem
######################
# The business problem is to build a hybrid movie recommender system using item-based and user-based
# approaches to provide personalized movie recommendations for users. The goal is to suggest top 10
# relevant films that align with their individual preferences.

# Dataset Stories
#########################
#   MovieLens is a film recommendation service that offers personalized movie suggestions to its users.
#   The dataset contains 2,000,263 ratings for 27,278 films.
#   The dataset was generated on October 17, 2016, and includes data collected from 138,493 unique users.
#   The data covers movie preferences between January 9, 1995, and March 31, 2015.
#   All users in the dataset were randomly selected.
#   Each user has rated a minimum of 20 films, resulting in a robust representation of diverse user tastes and interests.

# Variables:
#########################################################
# 1. movie.csv
#       movieId: Unique movie number (UniqueID).
#       title: Movie title.
#       genres: Genre(s) of the movie.
# 2. rating.csv
#       userid: Unique user number (UniqueID).
#       movieId: Unique movie number (UniqueID).
#       rating: Rating given by the user to the movie.
#       timestamp: Date of the rating.

###############################################################
# 2. Data Preparation
###############################################################

# Importing libraries
##############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Loading the datasets and merging them with 'movieId'
######################################################
movie_df = pd.read_csv("Datasets/movie.csv")
rating_df = pd.read_csv("Datasets/rating.csv")

df = movie_df.merge(rating_df, how="left", on="movieId")

df.shape
# The data includes 20000797 votes (ratings) in the rows and 6 variables in the columns.

# Total number of movies
##########################################
df["title"].nunique() # --> 27262 movies

# Number of ratings per movie
##############################
df["title"].value_counts() # --> while avg of first 5 is around 60k, last 5 is 1.

# Removing movies with a total rating count below 1000
#######################################################
# To do this, first, creating a dataframe that provides the total number of votes for each movie
comment_counts = pd.DataFrame(df["title"].value_counts())
# Than, assign the movie titles with less than 1000 ratings to a variable named "rare_movies"
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
rare_movies.shape # 24103 movies have less than 1000 ratings

# Last, creating a dataframe that contains the ratings of movies with 1000 or more ratings
common_movies_df = df[~df["title"].isin(rare_movies)]
common_movies_df.shape # --> 17766015 ratings
common_movies_df["title"].nunique() # --> 3159 movies

# Creating a pivot table for the DataFrame where "userID" values are in the index,
# movie titles are in the columns, and the ratings are used as values.
##################################################################################
user_movie_df = common_movies_df.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.shape # --> (138493, 3159) --> 138493 users, 3159 movies

# The function for data preparation - includes all the steps above.
###################################################################
def create_user_movie_df():
    import pandas as pd
    movie_df = pd.read_csv("Datasets/movie.csv")
    rating_df = pd.read_csv("Datasets/rating.csv")
    df = movie_df.merge(rating_df, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies_df = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies_df.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

###############################################################
# 3. User-Based Collaborative Filtering
###############################################################
# This method makes recommendations based on the similarity between users.
# Random selection of a user ID
###############################
random_user = int(pd.Series(user_movie_df.index).sample(1).values)
print(random_user) # --> UserID '117797'

# Creating a dataframe named 'random_user_df' consisting of observations
# belonging to the selected user
########################################################################
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.shape # --> I didnt choose a film so all films returned

# Assigning the movies that the selected user has rated, to a list named 'movies_watched'
########################################################################################
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched)# --> 21 films watched

# Creating a new dataframe that includes only the movies rated by the selected user from the 'user_movie_df' dataFrame
##############################################################
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.shape # --> (138493 users, 21 movies)

# Create a new dataFrame named 'user_movie_count' that contains information about
# how many of the films each user has watched, specifically the films
# watched by the selected user
##################################################################################################
user_movie_count = movies_watched_df.T.notnull().sum() # --> Count of films watched by each user
user_movie_count = user_movie_count.reset_index() # --> Creating dataframe
user_movie_count.columns = ["userId", "movie_count"] # --> Renaming column names

# Creating a list named 'users_same_movies' that includes the user IDs of users who have watched
# 60% or more of the films rated by the selected user
################################################################################################
perc = len(movies_watched) * 60/100 # --> 12.6 movies
users_same_moveis = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
users_same_moveis.shape  #--> 5486 users

# Filtering the 'movies_watched_df' dataframe to include the user IDs that exhibit
# similarity with the selected user in the 'user_same_movies'
#################################################################################
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_moveis)], random_user_df[movies_watched]])
final_df.shape # --> 5486 users - 21 movies

# Creating a new dataframe named 'corr_df' to calculate the correlations between users
######################################################################################
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates() # serie
corr_df = pd.DataFrame(corr_df, columns=["corr"]) # dataframe
corr_df.index.names = ['user_id_1', 'user_id_2'] # indexes
corr_df = corr_df.reset_index() # indexes --> variables
corr_df[corr_df["user_id_1"] == random_user] # The correlation of the randomly selected user with other users


# Creating a new dataframe named "top_users" by filtering users who have a high correlation (above 0.65) with the selected user
##################################################################
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True) # renaming variable as 'userId'

# The merging of the "top_users" dataframe with the "rating" dataset
####################################################################
rating = pd.read_csv("Datasets/rating.csv")
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]],how='inner') #to see the ratings given by users who have a high
# correlation withs the selected random user
top_users_ratings = top_users_ratings[top_users_ratings["userId"]!= random_user] # Excluding the random user from the dataset because the corr is 1


# Creating a new variable named "weighted_rating" by multiplying the correlation and rating values for each user
################################################################
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Creating a new dataframe named "recommendation_df" that contains the average weighted ratings
# for each movie ID, considering all users
###############################################################################################
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

# In the "recommendation_df," selecting films with a weighted rating greater than 3.5
# and sorting them based on the weighted rating in descending order
######################################################################################
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].\
    sort_values("weighted_rating", ascending=False).head(5)

# Fetching movie titles from the movie dataset and selecting the top 5 recommended films
#########################################################################################
movie = pd.read_csv('Datasets/movie.csv')
movies_to_be_recommend = movies_to_be_recommend.merge(movie[["movieId", "title"]])
print(movies_to_be_recommend)


###############################################################
# 4. Item-Based Collaborative Filtering
###############################################################
# Recommendations are made based on item similarity. (made by watched most recently rated)
# random_user_above = 117797

movie_df = pd.read_csv("Datasets/movie.csv")
rating_df = pd.read_csv("Datasets/rating.csv")
movies_to_be_recommend.merge(movie[["movieId", "title"]])
df = movie_df.merge(rating_df, how="left", on="movieId")

# Retrieving the movie ID of the most recently rated film from the movies that the user has given a rating of 5
##########################################################################
movies_user = df[(df['userId'] == 117797) & (df['rating'] == 5.0)]
movies_user = movies_user.sort_values(by='timestamp', ascending=False).head(1)
# movieId: 3 - title: That Thing You Do! (1996)

# Filtering the 'user_movie_df' dataFrame based on the selected movie title
############################################################################
filtered_df = user_movie_df['That Thing You Do! (1996)']

# Finding and ranking the correlation of the selected movie with other movies
#############################################################################
corrs = user_movie_df.corrwith(filtered_df).sort_values(ascending=False)

# Recommending the first 5 movies (excluding the selected movie itself)
#######################################################################
recommendations = corrs[corrs.index != 'That Thing You Do! (1996)'].sort_values(ascending=False).head(5)
print(recommendations)
# Top 5 recommendations
# Confidence (2003)
# Little Big League (1994)
# Miracle (2004)
# Doc Hollywood (1991)
# Easy A (2010)

