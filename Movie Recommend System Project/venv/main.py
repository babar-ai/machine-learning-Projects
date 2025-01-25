import pandas as pd
import json

#to import dataset
credits = pd.read_csv(r"F:\Machine Learning Hub\Machine Learning Projects\Movie Recommend System Project\TMDB 5000 Movie Dataset\tmdb_5000_credits.csv", delimiter=',', low_memory=False)
Movies = pd.read_csv(r"F:\Machine Learning Hub\Machine Learning Projects\Movie Recommend System Project\TMDB 5000 Movie Dataset\tmdb_5000_movies.csv", low_memory=False)

print('credits dataset shape: ',credits.shape)
print('Moivies dataset shape: ',Movies.shape)

# print(Movies.head(1))
print(Movies.iloc[0].genres)

for name in 
