# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Judul
st.title("\U0001F3D5️ Rekomendasi Wisata Magelang")
st.markdown("Sistem Rekomendasi Menggunakan Collaborative Filtering - Matrix Factorization Sederhana")

# Load data
@st.cache_data
def load_data():
    rating_df = pd.read_csv("Dataset_Rating_Mgl.csv")
    place_df = pd.read_csv("Dataset_tourisMagelang.csv")
    return rating_df, place_df

# Load model
@st.cache_data
def load_model():
    rating_matrix = joblib.load("rating_matrix.pkl")
    user_similarity_df = joblib.load("user_similarity.pkl")
    return rating_matrix, user_similarity_df

# Load all
rating_df, place_df = load_data()
rating_matrix, user_similarity_df = load_model()

# Fungsi prediksi rating
def predict_rating(user_id, place_id):
    if place_id in rating_matrix.columns:
        sim_scores = user_similarity_df.loc[user_id]
        ratings = rating_matrix[place_id]
        mask = ratings.notna() & (ratings.index != user_id)
        relevant_sims = sim_scores[mask]
        relevant_ratings = ratings[mask]
        if relevant_sims.sum() > 0:
            return np.dot(relevant_sims, relevant_ratings) / relevant_sims.sum()
    return None

# Fungsi rekomendasi
def recommend_places(user_id, top_n=3):
    user_ratings = rating_matrix.loc[user_id]
    unrated_places = user_ratings[user_ratings.isna()].index
    predictions = []
    for place_id in unrated_places:
        pred = predict_rating(user_id, place_id)
        if pred:
            place_name = place_df[place_df['Place_Id'] == place_id]['Place_Name'].values[0]
            predictions.append((place_name, pred))
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    return sorted_preds[:top_n]

# UI Pilih User
user_ids = rating_df['User_Id'].unique()
selected_user = st.selectbox("Pilih User ID:", user_ids)

if st.button("Tampilkan Rekomendasi"):
    rekomendasi = recommend_places(selected_user, top_n=3)
    if rekomendasi:
        st.subheader("\U0001F4C5 Rekomendasi Tempat Wisata:")
        for nama, skor in rekomendasi:
            st.markdown(f"**{nama}** — Prediksi Rating: `{skor:.2f}`")
    else:
        st.warning("User ini telah menilai semua tempat wisata.")

st.markdown("---")
st.caption("Dibuat oleh Armis Dayanti ❤️ dengan Streamlit")
