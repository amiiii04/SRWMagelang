import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

st.set_page_config(page_title="Rekomendasi Wisata Magelang", page_icon="🏞️", layout="wide")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    rating_df = pd.read_csv("Dataset_Rating_Mgl.csv")
    place_df = pd.read_csv("Dataset_tourisMagelang.csv")
    user_df = pd.read_csv("Dataset_usermgl.csv")
    return rating_df, place_df, user_df

# ===============================
# LOAD MODEL MANUAL SVD
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("mf_model_manual.pkl")

rating_df, place_df, user_df = load_data()
model = load_model()

pred_matrix = model["pred_matrix"]
users = model["users"]
places = model["places"]

# ===============================
# FUNGSI REKOMENDASI
# ===============================
def recommend_places(user_id, top_n=5):
    if user_id not in users:
        return []

    user_index = users.index(user_id)
    user_ratings = pred_matrix[user_index]

    place_scores = list(zip(places, user_ratings))
    place_scores.sort(key=lambda x: x[1], reverse=True)

    top_rekom = []
    for pid, score in place_scores:
        name = place_df[place_df["Place_Id"] == pid]["Place_Name"].values[0]
        top_rekom.append((name, score))

        if len(top_rekom) == top_n:
            break

    return top_rekom

# ===============================
# SEARCH WISATA
# ===============================
def search_place(keyword):
    keyword_lower = keyword.lower()

    name_match = place_df[
        place_df['Place_Name'].str.contains(keyword, case=False, na=False)
    ]

    desc_match = place_df[
        place_df['Description'].fillna('').str.contains(keyword, case=False, na=False)
    ]

    results = pd.concat([name_match, desc_match]).drop_duplicates()
    return results

# ===============================
# UI STREAMLIT
# ===============================
st.title("🏞️ Sistem Rekomendasi Wisata Magelang")
st.caption("Menggunakan Matrix Factorization (SVD Manual NumPy)")

st.markdown("---")

# ===============================
# SEARCH BAR
# ===============================
search_query = st.text_input("🔍 Cari Tempat Wisata...", placeholder="Misal: Borobudur")

if search_query:
    results = search_place(search_query)

    if results.empty:
        st.warning("Tidak ditemukan.")
    else:
        for _, row in results.iterrows():
            st.subheader(f"📍 {row['Place_Name']}")

            desc = row["Description"] if pd.notna(row["Description"]) else ""
            highlighted = re.sub(f"(?i)({search_query})", r"**\1**", desc)
            st.markdown(highlighted)

            avg = rating_df[rating_df["Place_Name"] == row["Place_Name"]]["Place_Rating"].mean()
            st.write(f"⭐ Rating rata-rata: {avg:.2f}")

            st.markdown("---")

# ===============================
# REKOMENDASI
# ===============================
st.subheader("🎯 Rekomendasi Berdasarkan User ID")

selected_user = st.selectbox("Pilih User ID:", users)

if st.button("Tampilkan Rekomendasi"):
    rekom = recommend_places(selected_user)

    for place, score in rekom:
        st.markdown(f"- **{place}** — Prediksi Rating `{score:.2f}` ⭐")

st.sidebar.info("Sistem Rekomendasi Wisata Magelang (SVD Manual)")
