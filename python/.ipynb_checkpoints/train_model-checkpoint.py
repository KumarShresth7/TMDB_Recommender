import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import ast

# Load and preprocess data
movies = pd.read_csv('data/tmdb_5000_movies.csv')
cred = pd.read_csv('data/tmdb_5000_credits.csv')

movies = movies.merge(cred, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# Helper functions for preprocessing
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def convert3(text):
    return [i['name'] for i in ast.literal_eval(text)[:3]]

def fetch_director(text):
    return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)

# Process text fields for tags
for col in ['overview', 'genres', 'keywords', 'cast', 'crew']:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# Save DataFrame for consistency
pickle.dump(new_df, open('model/movies.pkl', 'wb'))

# Vectorization and similarity computation
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vector)

# Save CountVectorizer and similarity matrix
pickle.dump(cv, open('model/count_vectorizer.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))

# Recommendation function using CountVectorizer
def recommend(movie):
    try:
        movie_index = new_df[new_df['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        return [new_df.iloc[i[0]].title for i in movies_list]
    except IndexError:
        return ["Movie not found."]
print("Movie Recommendation using Content-Based Model: ")
print(recommend('Avatar'))

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(new_df['tags']).toarray()
tfidf_similarity = cosine_similarity(tfidf_matrix)

# Save TF-IDF vectorizer and matrix
pickle.dump(tfidf, open('model/tfidf_vectorizer.pkl', 'wb'))
pickle.dump(tfidf_similarity, open('model/tfidf_similarity.pkl', 'wb'))

# TF-IDF Recommendation function
def recommend_tfidf(movie):
    try:
        movie_index = new_df[new_df['title'] == movie].index[0]
        distances = tfidf_similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        return [new_df.iloc[i[0]].title for i in movies_list]
    except IndexError:
        return ["Movie not found."]
print("Movie Recommendation using TF-IDF Model: ")
print(recommend_tfidf('Avatar'))

# K-Means clustering with saved vector space
kmeans = KMeans(n_clusters=20, random_state=42)
new_df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Save KMeans model
pickle.dump(kmeans, open('model/kmeans_model.pkl', 'wb'))

def recommend_kmeans(movie):
    try:
        movie_index = new_df[new_df['title'] == movie].index[0]
        movie_cluster = new_df.iloc[movie_index]['cluster']
        similar_movies = new_df[new_df['cluster'] == movie_cluster].sort_values('title')
        recommendations = similar_movies['title'].tolist()
        if movie in recommendations:
            recommendations.remove(movie)
        return recommendations[:5]
    except IndexError:
        return ["Movie not found."]
print("Movie Recommendation using KMeans Model: ")
print(recommend_kmeans('Avatar'))

# Evaluation
def evaluate_model(recommend_func, test_movies):
    precision_scores, recall_scores, diversity_scores = [], [], []

    def calculate_diversity(recommendations):
        return len(set(recommendations)) / len(recommendations) if recommendations else 0

    for movie in test_movies:
        recommended = recommend_func(movie)
        if not recommended:
            continue
        relevance = set(recommended[:3])  # Assume top-3 are relevant
        precision = len(relevance) / len(recommended) if recommended else 0
        recall = len(relevance) / 3  # Total assumed relevant is 3
        diversity = calculate_diversity(recommended)

        precision_scores.append(precision)
        recall_scores.append(recall)
        diversity_scores.append(diversity)

    return {
        "Precision": np.mean(precision_scores),
        "Recall": np.mean(recall_scores),
        "Diversity": np.mean(diversity_scores)
    }

# Test and visualize results
test_movies = ['Avatar', 'Titanic', '12 Rounds', 'Inception', 'Interstellar']
content_results = evaluate_model(recommend, test_movies)
tfidf_results = evaluate_model(recommend_tfidf, test_movies)
kmeans_results = evaluate_model(recommend_kmeans, test_movies)

models = ['Content-Based', 'TF-IDF', 'K-Means']
precisions = [content_results['Precision'], tfidf_results['Precision'], kmeans_results['Precision']]
recalls = [content_results['Recall'], tfidf_results['Recall'], kmeans_results['Recall']]
diversities = [content_results['Diversity'], tfidf_results['Diversity'], kmeans_results['Diversity']]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, precisions, width, label='Precision')
ax.bar(x, recalls, width, label='Recall')
ax.bar(x + width, diversities, width, label='Diversity')

ax.set_ylabel('Scores')
ax.set_title('Comparison of Models')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.show()

# K-Means cluster visualization
plt.figure(figsize=(10, 6))
plt.hist(new_df['cluster'], bins=20, color='skyblue', edgecolor='black')
plt.title('Cluster Distribution in K-Means')
plt.xlabel('Cluster')
plt.ylabel('Number of Movies')
plt.show()

import seaborn as sns

# Generate a heatmap for a subset of similarity matrix (first 20 movies)
subset_similarity = similarity[:20, :20]  # Adjust the subset size as needed
plt.figure(figsize=(12, 8))
sns.heatmap(subset_similarity, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=new_df['title'][:20], yticklabels=new_df['title'][:20])
plt.title("Heatmap of Cosine Similarity (Subset)")
plt.xticks(rotation=45)
plt.show()




