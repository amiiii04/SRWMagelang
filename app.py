import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re

# ===============================
# 🔧 KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Rekomendasi Wisata Magelang",
    page_icon="🏞️",
    layout="wide"
)


# ===============================
# 📂 DEBUG: CEK FILE YANG ADA
# ===============================
st.sidebar.subheader("📁 Debugging Files")
st.sidebar.write("Files in working directory:")
st.sidebar.write(os.listdir("."))


# ===============================
# 📂 LOAD DATASET
# ===============================
@st.cache_data
def load_data():
    try:
        rating_df = pd.read_csv("Dataset_Rating_Mgl.csv")
        place_df = pd.read_csv("Dataset_tourisMagelang.csv")
        user_df = pd.read_csv("Dataset_usermgl.csv")
    except FileNotFoundError:
        st.error("❌ File dataset tidak ditemukan. Pastikan file CSV berada di folder yang sama dengan app.py")
        return None, None, None
    return rating_df, place_df, user_df


# ===============================
# 📦 LOAD MODEL SVD / MF
# ===============================
@st.cache_resource
def load_model():

    # Debug informasi folder
    st.write("📂 Current Working Directory:", os.getcwd())
    st.write("📄 Files:", os.listdir("."))

    model_path = "mf_model.pkl"

    if not os.path.exists(model_path):
        st.error(f"❌ File model SVD '{model_path}' tidak ditemukan.")
        st.info("💡 Pastikan file mf_model.pkl berada di direktori yang sama dengan app.py.")
        return None

    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Gagal load model. Error: {e}")
        return None

    return model


rating_df, place_df, user_df = load_data()
model = load_model()

# Jika dataset atau model tidak tersedia → berhenti
if rating_df is None or model is None:
    st.stop()


# ===============================
# 🧠 FUNGSI PREDIKSI (SVD/MF)
# ===============================
def predict_rating(user_id, place_id):
    try:
        pred = model.predict(user_id, place_id)
        return pred.est
    except Exception as e:
        st.error(f"❌ Gagal memprediksi: {e}")
        return 0


# ===============================
# ⭐ FUNGSI REKOMENDASI
# ===============================
def recommend_places(user_id, top_n=5):
    rated_places = rating_df[rating_df['User_Id'] == user_id]['Place_Id'].values
    all_places = place_df['Place_Id'].values
    unrated_places = [p for p in all_places if p not in rated_places]

    predictions = []
    for place in unrated_places:
        est = predict_rating(user_id, place)
        place_name = place_df[place_df['Place_Id'] == place]['Place_Name'].values[0]
        predictions.append((place_name, est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]


# ===============================
# 🔍 FUNGSI SEARCH WISATA
# ===============================
def search_place(keyword):
    keyword_lower = keyword.lower()

    name_match = place_df[place_df['Place_Name'].str.contains(keyword, case=False, na=False)].copy()
    desc_match = place_df[place_df['Description'].str.contains(keyword, case=False, na=False)].copy()

    results = pd.concat([name_match, desc_match]).drop_duplicates().reset_index(drop=True)
    
    if results.empty:
        return results

    def relevance_score(row):
        name_score = row['Place_Name'].lower().count(keyword_lower)
        desc_score = row['Description'].lower().count(keyword_lower)
        return name_score * 2 + desc_score

    results["Relevance"] = results.apply(relevance_score, axis=1)
    return results.sort_values("Relevance", ascending=False)


# ===============================
# 🖥️ UI STREAMLIT
# ===============================
st.title("🏞️ Sistem Rekomendasi Wisata Magelang")
st.caption("Menggunakan *Collaborative Filtering* dengan algoritma *Matrix Factorization (SVD)*")

st.markdown("---")


# ===============================
# 🔍 SEARCH BAR
# ===============================
search_query = st.text_input("🔍 Cari Tempat Wisata", placeholder="Misal: Borobudur")

if search_query:
    results = search_place(search_query)

    if results.empty:
        st.warning("❌ Tempat tidak ditemukan.")
    else:
        for idx, row in results.iterrows():
            st.subheader(f"📍 {row['Place_Name']}")

            desc = row['Description']
            highlighted_desc = re.sub(f"(?i)({search_query})", r"**\\1**", desc)
            st.markdown(f"📝 {highlighted_desc}")

            avg = rating_df[rating_df['Place_Name'] == row['Place_Name']]['Place_Rating'].mean()
            st.write(f"⭐ Rata-rata Rating: {avg:.2f}/5.0")

            st.markdown("---")
else:
    st.info("🔎 Cari tempat wisata untuk melihat detailnya.")


# ===============================
# 🎯 REKOMENDASI BERDASARKAN USER
# ===============================
st.subheader("🎯 Rekomendasi Berdasarkan User ID")

selected_user = st.selectbox(
    "Pilih User ID:",
    sorted(user_df['User_Id'].unique())
)

if st.button("Tampilkan Rekomendasi"):
    rekom = recommend_places(selected_user, top_n=5)

    st.write(f"Top 5 rekomendasi untuk User {selected_user}:")
    for place, score in rekom:
        st.markdown(f"- **{place}** — Prediksi Rating: `{score:.2f}` ⭐")

st.sidebar.caption("✨ Sistem Rekomendasi Wisata Magelang — SVD Version")
