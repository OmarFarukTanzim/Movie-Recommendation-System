🎬 Movie Recommendation System</br>
 A deep learning-based movie recommendation system built using PyTorch and deployed with Streamlit.
 The system uses a Hybrid Neural Collaborative Filtering (NCF) model to predict user preferences and recommend movies.

🚀 Features


Personalized recommendations using user ID


Hybrid approach (collaborative + content-based filtering)


Uses movie genres as additional features


Predicts ratings on a 0–5 scale


Handles cold-start users (popular movies fallback)


Interactive web app using Streamlit


Displays movie posters using TMDB API



📂 Project Structure</br>
├── app.py                      # Streamlit application </br>
├── Movie_Recommendation.ipynb  # Model training notebook </br>
├── best_recommender.pth        # Trained model weights </br>
├── ratings.csv                 # User ratings dataset </br>
├── movies.csv                  # Movie metadata </br>
├── links.csv                   # External ID mappings </br>
├── README.md

📊 Dataset </br>
This project uses three datasets: </br>


movies.csv : Contains movie information </br>
movie_id, title, genres


ratings.csv : Contains user ratings </br>
user_id, movie_id, rating, timestamp


links.csv : Maps movie IDs to external databases </br>
movie_id, imdb_id, tmdb_id


Preprocessing


Merged datasets using movie_id


Encoded user_id and movie_id


Converted genres into one-hot vectors


Prepared final dataset for training



🧠 Model
Hybrid Neural Collaborative Filtering (NCF):


User embedding


Movie embedding


Genre feature layer


Fully connected layers: 128 → 64 → 32


Dropout for regularization


User & movie bias terms


Output: predicted rating (0–5)



⚙️ Training


Loss Function: Mean Squared Error (MSE)


Optimizer: Adam


Train/Test Split: 80/20



🔍 Recommendation Logic


Input user ID


Remove already watched movies


Filter unpopular movies (minimum rating threshold)


Predict ratings for unseen movies


Rank movies by predicted score


Return top recommendations



🌐 Run the Project
1. Clone the repository
git clone https://github.com/your-username/movie-recommender.gitcd movie-recommender
2. Install dependencies
pip install -r requirements.txt
3. Run the app
streamlit run app.py

🔑 API Setup
This project uses TMDB API to fetch movie posters.
Replace the API key in app.py:
api_key = "YOUR_API_KEY"



👨‍💻 Author</br>
Omar Faruk Tanzim </br>
Information & Communication Engineering </br>
University of Rajshahi

