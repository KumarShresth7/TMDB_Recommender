import pickle
import streamlit as st
import requests
import pandas as pd

# Load the required files
st.title('Movie Recommender System')

movies_dict = pickle.load(open('model/movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('model/similarity.pkl', 'rb'))
tfidf_similarity = pickle.load(open('model/tfidf_similarity.pkl', 'rb'))  # Load the precomputed TF-IDF similarity
kmeans_model = pickle.load(open('model/kmeans_model.pkl', 'rb'))  # Load the pre-trained KMeans model

def recommend(movie, model_type):
    movie_index = movies[movies['title'] == movie].index[0]
    recommended_movies = []
    recommended_movies_posters = []
    recommended_movie_trailers = []  # List to store trailer links

    if model_type == "Content-Based":
        distances = similarity[movie_index]
    elif model_type == "TF-IDF":
        distances = tfidf_similarity[movie_index]
    elif model_type == "KMeans":
        movie_cluster = movies.iloc[movie_index]['cluster']
        similar_movies = movies[movies['cluster'] == movie_cluster].sort_values('title')
        for idx, row in similar_movies.iterrows():
            if row['title'] != movie:
                recommended_movies.append(row['title'])
                recommended_movies_posters.append(fetch_poster(row['movie_id']))
                recommended_movie_trailers.append(fetch_trailer(row['movie_id']))  # Fetch trailer link
            if len(recommended_movies) >= 5:  # Limit to 5 recommendations
                break
        return recommended_movies, recommended_movies_posters, recommended_movie_trailers

    # If using Content-Based or TF-IDF, sort by similarity score
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
        recommended_movie_trailers.append(fetch_trailer(movie_id))  # Fetch trailer link
        
    return recommended_movies, recommended_movies_posters, recommended_movie_trailers  # Return trailer links

def fetch_poster(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def fetch_trailer(movie_id):
    response = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    )
    data = response.json()
    for video in data.get("results", []):
        if video["type"] == "Trailer":
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None

# Add custom CSS
st.markdown(
    """
    <style>
    .stImage {
        padding: 10px;  /* Adjust the padding as needed */
    }
    .trailer-link {
        color: #0073e6;
        font-weight: bold;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User input for model selection
selected_movie_name = st.selectbox(
    'Select a movie:',
    movies['title'].values
)

selected_model = st.selectbox(
    'Select Recommendation Model:',
    ['Content-Based', 'TF-IDF', 'KMeans']  # Added KMeans to model options
)

# Display columns with posters and trailers
if st.button('Show Recommendations'):
    recommended_movie_names, recommended_movie_posters, recommended_movie_trailers = recommend(selected_movie_name, selected_model)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0], use_column_width=True)
        if recommended_movie_trailers[0]:
            st.markdown(f'<a href="{recommended_movie_trailers[0]}" target="_blank" class="trailer-link"><i class="fa fa-play"></i> Watch Trailer</a>', unsafe_allow_html=True)
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1], use_column_width=True)
        if recommended_movie_trailers[1]:
            st.markdown(f'<a href="{recommended_movie_trailers[1]}" target="_blank" class="trailer-link"><i class="fa fa-play"></i> Watch Trailer</a>', unsafe_allow_html=True)
    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2], use_column_width=True)
        if recommended_movie_trailers[2]:
            st.markdown(f'<a href="{recommended_movie_trailers[2]}" target="_blank" class="trailer-link"><i class="fa fa-play"></i> Watch Trailer</a>', unsafe_allow_html=True)
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3], use_column_width=True)
        if recommended_movie_trailers[3]:
            st.markdown(f'<a href="{recommended_movie_trailers[3]}" target="_blank" class="trailer-link"><i class="fa fa-play"></i> Watch Trailer</a>', unsafe_allow_html=True)
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4], use_column_width=True)
        if recommended_movie_trailers[4]:
            st.markdown(f'<a href="{recommended_movie_trailers[4]}" target="_blank" class="trailer-link"><i class="fa fa-play"></i> Watch Trailer</a>', unsafe_allow_html=True)
