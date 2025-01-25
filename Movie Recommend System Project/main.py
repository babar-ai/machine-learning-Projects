#The crew column typically contains information about the people involved in the production of the movie other than the cast. This includes directors, writers, producers, cinematographers, etc.
#The cast column typically contains information about the actors and actresses who appear in the movie.


import pandas as pd
import ast 


#to load dataset
credits = pd.read_csv(r"F:\Machine Learning Hub\Machine Learning Projects\Movie Recommend System Project\TMDB 5000 Movie Dataset\tmdb_5000_credits.csv",low_memory=False)
movies = pd.read_csv(r"F:\Machine Learning Hub\Machine Learning Projects\Movie Recommend System Project\TMDB 5000 Movie Dataset\tmdb_5000_movies.csv",low_memory=False)


#to concatinate/merge both datasets
movies = movies.merge(credits, on='title')

#drop null values
movies.dropna(inplace=True)

#drop duplicate values
movies.drop_duplicates(inplace=True)


#filterout specific columns which helps us in creating tags
movies = movies[['movie_id', 'title', 'cast', 'crew', 'genres', 'overview','keywords']]


def Convert(obj):
    
    characters = []
    for i in ast.literal_eval(obj):
        characters.append(i['name'])
    
    return characters

def Convert2(obj):
    characters = []
    count = 0
    for i in ast.literal_eval(obj):
        if count == 3:
            break
        else:
            characters.append(i['name'])
            count += 1
            
    return characters
        


#for fetching data from column "name" where job = director
def fetch_director(obj):
    directors = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            directors.append(i['name'])
            break
    
    return directors
        

movies['cast'] = movies['cast'].apply(Convert2)
movies['genres'] = movies['genres'].apply(Convert)
movies['keywords'] = movies['keywords'].apply(Convert)
movies['crew'] = movies['crew'].apply(fetch_director)
print('success')

#to convert "overview" column into list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# print(movies['crew'])
# print(movies['genres'])
# print(movies['keywords'])
# print(movies['cast'])
# print(movies['overview'])
# print(movies.head(1))

#NOW 
# AS we have applied datapreproccessing successfully on dataset and have convert all the column into list format

#to remove spaces amoung each list item i.e name = "babar raheem" we have to convert it into "babarraheem"
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])
print('success2')


