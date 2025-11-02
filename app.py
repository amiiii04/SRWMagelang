import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==== KONFIGURASI HALAMAN ====
st.set_page_config(
    page_title="Rekomendasi Wisata Magelang",
    page_icon="🏞️",  # Emoji sebagai ikon
    layout="wide"  # Menggunakan layout lebar
)

# ==== LOAD DATA & MODEL (Menggunakan cache agar cepat) ====
@st.cache_data
def load_data():
    try:
        rating_df = pd.read_csv("Dataset_Rating_Mgl.csv")
        place_df = pd.read_csv("Dataset_tourisMagelang.csv")
    except FileNotFoundError:
        st.error("Error: File dataset tidak ditemukan. Pastikan 'Dataset_Rating_Mgl.csv' dan 'Dataset_tourisMagelang.csv' ada di folder yang sama.")
        return None, None
    return rating_df, place_df

@st.cache_data
def load_model():
    try:
        rating_matrix = joblib.load("rating_matrix.pkl")
        user_similarity_df = joblib.load("user_similarity.pkl")
    except FileNotFoundError:
        st.error("Error: File model tidak ditemukan. Pastikan 'rating_matrix.pkl' dan 'user_similarity.pkl' ada.")
        return None, None
    return rating_matrix, user_similarity_df

# Load all
rating_df, place_df = load_data()
rating_matrix, user_similarity_df = load_model()

# Cek jika data gagal di-load
if rating_df is None or place_df is None or rating_matrix is None or user_similarity_df is None:
    st.stop()  # Menghentikan eksekusi script jika file penting tidak ada

# ==== FUNGSI REKOMENDASI (Sedikit dimodifikasi) ====
def predict_rating(user_id, place_id):
    if place_id in rating_matrix.columns:
        sim_scores = user_similarity_df.loc[user_id]
        ratings = rating_matrix[place_id]
        
        # Filter data: bukan user_id itu sendiri dan yang sudah ada ratingnya
        mask = ratings.notna() & (ratings.index != user_id)
        relevant_sims = sim_scores[mask]
        relevant_ratings = ratings[mask]
        
        if relevant_sims.sum() > 0:
            # Perhitungan weighted average
            return np.dot(relevant_sims, relevant_ratings) / relevant_sims.sum()
    return None # Return None jika tidak bisa diprediksi

def recommend_places(user_id, top_n=3):
    user_ratings = rating_matrix.loc[user_id]
    # Dapatkan daftar place_id yang 'belum' dinilai oleh user
    unrated_places = user_ratings[user_ratings.isna()].index
    
    predictions = []
    for place_id in unrated_places:
        pred = predict_rating(user_id, place_id)
        if pred:
            # Dapatkan semua info tempat wisata dari place_df
            place_info = place_df[place_df['Place_Id'] == place_id].iloc[0]
            predictions.append((place_info, pred)) # Simpan (info_lengkap, skor_prediksi)
            
    # Urutkan berdasarkan skor prediksi (tertinggi ke terendah)
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    return sorted_preds[:top_n]

def get_user_rated_history(user_id):
    user_rated_items = rating_matrix.loc[user_id].dropna()
    if user_rated_items.empty:
        return []
    
    rated_places = []
    for place_id, rating in user_rated_items.items():
        place_info = place_df.loc[place_df['Place_Id'] == place_id]
        if not place_info.empty:
            place_name = place_info['Place_Name'].values[0]
            rated_places.append((place_name, rating))
    return rated_places

# ==== UI: SIDEBAR ====
st.sidebar.title("👤 Panel Pengguna")
user_ids = rating_df['User_Id'].unique()
selected_user = st.sidebar.selectbox("Pilih User ID:", user_ids)

st.sidebar.info(f"Menampilkan rekomendasi untuk **User {selected_user}**.")
st.sidebar.caption("Dibuat oleh Armis Dayanti ❤️")

# ==== UI: HALAMAN UTAMA ====

# Judul Utama
st.title("🏞️ Rekomendasi Wisata Magelang")
st.markdown("Sistem Rekomendasi Menggunakan *Collaborative Filtering*")
st.markdown("---")

# Tampilkan Rekomendasi (TANPA TOMBOL, langsung update)
rekomendasi = recommend_places(selected_user, top_n=3)

st.header(f"✨ Rekomendasi Teratas untuk Anda (User {selected_user})")

if rekomendasi:
    # Buat 3 kolom untuk 3 rekomendasi
    cols = st.columns(3)
    
    for i, (place_info, skor) in enumerate(rekomendasi):
        with cols[i]:
            # ** ASUMSI KOLO M 'Image_URL' ADA **
            if 'Image_URL' in place_info and pd.notna(place_info['Image_URL']):
                st.image(place_info['Image_URL'], caption=f"Kategori: {place_info.get('Category', 'N/A')}")
            else:
                st.image("https://via.placeholder.com/400x300.png?text=Gambar+Tidak+Tersedia", caption="Gambar tidak tersedia")
            
            st.subheader(place_info['Place_Name'])
            st.markdown(f"**🌟 Prediksi Rating: {skor:.2f} / 5.0**")
            
            # ** ASUMSI KOLO M 'Description' ADA **
            if 'Description' in place_info:
                with st.container(height=120): # Batasi tinggi deskripsi
                    st.write(place_info['Description'])
else:
    st.warning("User ini telah menilai semua tempat wisata. Tidak ada rekomendasi baru.")

st.markdown("---")

# Bagian Bonus: Menampilkan riwayat rating user
st.header(f"⭐ Riwayat Rating Anda")
rated_history = get_user_rated_history(selected_user)

if rated_history:
    with st.expander("Lihat tempat yang sudah Anda nilai"):
        for nama, rating in rated_history:
            st.markdown(f"* **{nama}**: Anda memberi `{int(rating)}` ⭐")
else:
    st.info("Anda (User {selected_user}) belum memberikan rating pada tempat manapun.")
