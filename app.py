import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")

    movies['overview'] = movies['overview'].fillna('')
    movies['genres'] = movies['genres'].fillna('')
    movies['keywords'] = movies['keywords'].fillna('')

    movies['tags'] = (
        movies['overview'] +
        movies['genres'] +
        movies['keywords']
    )

    movies['tags'] = movies['tags'].str.lower().str.replace(" ", "")
    return movies


# -------------------- MODEL BUILDING --------------------
@st.cache_resource
def build_model(movies):
    cv = CountVectorizer(max_features=3000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity


movies = load_data()
similarity = build_model(movies)


# -------------------- RECOMMEND FUNCTION --------------------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i[0]].title for i in movies_list]


# -------------------- STREAMLIT UI --------------------
st.title("🎬 AI Movie Recommendation System")

movie_name = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend"):
    for m in recommend(movie_name):
        st.write(m)
