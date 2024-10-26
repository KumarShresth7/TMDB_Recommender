import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds 
import pickle

movies = pd.read_csv('data/tmdb_5000_movies.csv')
cred = pd.read_csv('data/tmdb_5000_credits.csv')

movies = movies.merge(cred,on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

import ast
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

#for getting top3 actors name only 
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter<3:
            L.append(i['name'])
        counter+=1
    return L

movies['cast'] = movies['cast'].apply(convert3)

#for getting director
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 

movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
# print(new_df['tags'].head())

#Data Preprocessing done upto here

#Vectorization begins from this point
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()
# print(vector)

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

def recommend(movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

print(recommend('Avatar'))

#create pickle model
pickle.dump(new_df,open('model/movies.pkl','wb'))

pickle.dump(new_df.to_dict(),open('model/movie_dict.pkl','wb'))
pickle.dump(similarity,open('model/similarity.pkl','wb'))
    
        





        


















