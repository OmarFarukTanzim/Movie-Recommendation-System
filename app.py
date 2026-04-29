import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

# ==========================================
# 1. Page configuration
# ==========================================
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.markdown("""
    <style>
    /* Dark mode background */
    .stApp {
        background-color: #141414;
        color: #FFFFFF;
    }
    
    /* App Title - Netflix Red */
    h1 {
        color: #E50914 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Movie Card Container */
    .movie-card {
        background-color: #222222;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
        overflow: hidden; /* Keeps the image inside the rounded corners */
        transition: transform 0.3s ease;
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    
    /* Hover effect to pop the card out */
    .movie-card:hover {
        transform: scale(1.05);
        border: 1px solid #E50914;
    }
    
    /* Movie Poster Image */
    .movie-poster {
        width: 100%;
        height: 300px;
        object-fit: cover; /* Prevents image stretching */
    }
    
    /* Info area below the poster */
    .movie-info {
        padding: 15px;
    }
    
    /* Movie Title */
    .movie-title {
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 5px;
        color: #FFFFFF;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis; /* Adds ... if title is too long */
    }
    
    /* Star Rating */
    .star-rating {
        color: #FFD700;
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING 
# ==========================================
@st.cache_data
def load_data():
    # Load your files (Ensure these match your actual filenames in the folder!)
    ratings = pd.read_csv('ratings.csv') 
    movies_encoded = pd.read_csv('movies.csv') 
    
    # Check if we need to do one-hot encoding on the fly
    if 'genres' in movies_encoded.columns and 'Action' not in movies_encoded.columns:
        genre_dummies = movies_encoded['genres'].str.get_dummies(sep='|')
        movies_encoded = pd.concat([movies_encoded, genre_dummies], axis=1)
    
    # Standard MovieLens 20 Genres
    possible_genres = [
        'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
        'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 
        'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)'
    ]
    
    genre_cols = [col for col in possible_genres if col in movies_encoded.columns]
    
    user2idx = {user: idx for idx, user in enumerate(ratings['userId'].unique())}
    movie2idx = {movie: idx for idx, movie in enumerate(ratings['movieId'].unique())}
    
    num_users = len(user2idx)
    num_movies = len(movie2idx)
    num_genres = len(genre_cols) 
    
    return ratings, movies_encoded, user2idx, movie2idx, genre_cols, num_users, num_movies, num_genres

ratings, movies_encoded, user2idx, movie2idx, genre_cols, num_users, num_movies, num_genres = load_data()

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class HybridNCF(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_dim=50, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        super(HybridNCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        
        self.genre_dense = nn.Linear(num_genres, embedding_dim) 
        
        input_size = embedding_dim * 3
        
        layers = []
        for i, h_size in enumerate(hidden_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_layers[i-1], h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate)) 
        self.hidden_layers = nn.Sequential(*layers)
        
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

    def forward(self, user_ids, movie_ids, genres):
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)
        
        u_bias = self.user_bias(user_ids).squeeze(1)
        m_bias = self.movie_bias(movie_ids).squeeze(1)
        
        genre_features = F.relu(self.genre_dense(genres))
        
        combined_features = torch.cat((user_embed, movie_embed, genre_features), dim=1)
        x = self.hidden_layers(combined_features)
        
        base_prediction = self.output_layer(x).squeeze(1)
        final_prediction = base_prediction + u_bias + m_bias
        
        prediction = torch.sigmoid(final_prediction) * 5
        return prediction

# ==========================================
# 4. LOAD THE TRAINED WEIGHTS
# ==========================================
@st.cache_resource
def load_model():
    model = HybridNCF(num_users, num_movies, num_genres, embedding_dim=50)
    model.load_state_dict(torch.load('best_recommender.pth', map_location=torch.device('cpu')))
    model.eval() 
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model. Details: {e}")

# ==========================================
# 5. POSTER FETCHING & INFERENCE LOGIC
# ==========================================
@st.cache_data
def fetch_poster(movie_title):
    # REPLACE WITH YOUR ACTUAL TMDB API KEY
    api_key = "Paste your key" 
    
    search_title = movie_title.split(' (')[0]
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={search_title}"
    
    try:
        response = requests.get(url).json()
        if response['results']:
            poster_path = response['results'][0]['poster_path']
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    
    return "https://via.placeholder.com/500x750/222222/E50914?text=No+Poster"

def get_recommendations(user_id, top_n=5, min_ratings=25):
    if user_id not in user2idx:
        return None 
    
    user_idx = user2idx[user_id]
    watched_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    
    movie_counts = ratings['movieId'].value_counts()
    popular_enough_movies = movie_counts[movie_counts >= min_ratings].index.tolist()
    
    unwatched_movies = movies_encoded[
        (~movies_encoded['movieId'].isin(watched_movie_ids)) & 
        (movies_encoded['movieId'].isin(movie2idx.keys())) &
        (movies_encoded['movieId'].isin(popular_enough_movies)) 
    ].copy()
    
    if unwatched_movies.empty:
        return None 
        
    movie_indices = torch.tensor([movie2idx[m_id] for m_id in unwatched_movies['movieId']], dtype=torch.long)
    user_indices = torch.tensor([user_idx] * len(movie_indices), dtype=torch.long)
    genre_tensors = torch.tensor(unwatched_movies[genre_cols].values, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(user_indices, movie_indices, genre_tensors)
        
    unwatched_movies['Predicted_Rating'] = predictions.numpy()
    
    top_movies = unwatched_movies.sort_values(by='Predicted_Rating', ascending=False).head(top_n)
    return top_movies[['movieId', 'title', 'genres', 'Predicted_Rating']]

# ==========================================
# 6. STREAMLIT UI & INTERACTION
# ==========================================
st.markdown("<h1>Movie Recommendation System</h1>", unsafe_allow_html=True)
st.write("Enter a User ID below to see what our Neural Network predicts they will love!")

user_input = st.number_input("User ID (e.g., 1 to 600):", min_value=1, step=1)

if st.button("Get Recommendations"):
    with st.spinner("Calculating neural embeddings..."):
        top_movies = get_recommendations(user_input, top_n=5, min_ratings=25)
        
        if top_movies is not None and not top_movies.empty:
            st.markdown(f"### Top Picks For User {user_input}")
            
            cols = st.columns(5)
            
            for i, row in enumerate(top_movies.itertuples()):
                with cols[i]:
                    poster_url = fetch_poster(row.title)
                    st.markdown(f"""
                        <div class="movie-card">
                            <img src="{poster_url}" class="movie-poster" alt="{row.title}">
                            <div class="movie-info">
                                <div class="movie-title" title="{row.title}">{row.title}</div>
                                <div class="star-rating">⭐ {row.Predicted_Rating:.2f}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("User not found or no recommendations available.")