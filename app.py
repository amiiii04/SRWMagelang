import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==== KONFIGURASI HALAMAN ====
st.set_page_config(
    page_title="Rekomendasi Wisata Magelang",
    page_icon="🏞️",
    layout="wide"
)

# ==== LOAD DATA & MODEL ====
@st.cache_data
def load_data():
    rating_df = pd.read_csv("Dataset_Rating_Mgl.csv")
    place_df = pd.read_csv("Dataset_tourisMagelang.csv")
    user_df = pd.read_csv("Dataset_usermgl.csv")
    return rating_df, place_df, user_df

@st.cache_data
def load_model():
    rating_matrix = joblib.load("rating_matrix.pkl")
    user_similarity_df = joblib.load("user_similarity.pkl")
    return rating_matrix, user_similarity_df

rating_df, place_df, user_df = load_data()
rating_matrix, user_similarity_df = load_model()

# ==== CEK DATA ====
if rating_df is None or place_df is None or rating_matrix is None or user_similarity_df is None:
    st.error("Dataset atau model tidak ditemukan. Pastikan semua file .csv dan .pkl ada di folder yang sama.")
    st.stop()

# ==== FUNGSI ULASAN OTOMATIS BERDASARKAN RATING ====
def generate_review_text(rating):
    if rating >= 5:
        return "Pengalaman luar biasa, sangat direkomendasikan!"
    elif rating >= 4:
        return "Tempatnya bagus, cukup memuaskan."
    elif rating >= 3:
        return "Biasa saja, tapi cukup oke untuk dikunjungi."
    elif rating >= 2:
        return "Kurang sesuai harapan."
    else:
        return "Tidak direkomendasikan."

# ==== FUNGSI REKOMENDASI ====
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

def recommend_places(user_id, top_n=3):
    user_ratings = rating_matrix.loc[user_id]
    unrated_places = user_ratings[user_ratings.isna()].index
    predictions = []
    for place_id in unrated_places:
        pred = predict_rating(user_id, place_id)
        if pred:
            place_info = place_df[place_df['Place_Id'] == place_id].iloc[0]
            predictions.append((place_info, pred))
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

# ==== SIDEBAR: PANEL USER ====
st.sidebar.title("👤 Panel Pengguna")
user_ids = rating_df['User_Id'].unique()
selected_user = st.sidebar.selectbox("Pilih User ID:", user_ids)

st.sidebar.info(f"Menampilkan rekomendasi untuk **User {selected_user}**.")
st.sidebar.caption("Dibuat oleh Armis Dayanti ❤️")

# ==== FITUR PENCARIAN TEMPAT WISATA ====
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Cari Tempat Wisata")
search_query = st.sidebar.text_input("Masukkan nama tempat wisata:")

if search_query:
    search_results = place_df[place_df['Place_Name'].str.contains(search_query, case=False, na=False)]
    if not search_results.empty:
        for _, row in search_results.iterrows():
            st.header(f"📍 {row['Place_Name']}")
            if 'Image_URL' in row and pd.notna(row['Image_URL']):
                st.image(row['Image_URL'], caption=row.get('Category', 'Kategori tidak tersedia'))
            st.write(f"**Kategori:** {row.get('Category', 'N/A')}")
            st.write(row.get('Description', 'Deskripsi tidak tersedia.'))
            
            # ==== TAMPILKAN ULASAN DARI PENGGUNA ====
            st.subheader("💬 Ulasan Pengguna Sebelumnya:")
            place_reviews = rating_df[rating_df['Place_Id'] == row['Place_Id']]
            if not place_reviews.empty:
                for _, review in place_reviews.iterrows():
                    user_name = user_df[user_df['User_Id'] == review['User_Id']]['User_Name'].values[0]
                    st.markdown(f"- **{user_name}** ({int(review['Place_Rating'])} ⭐): _{generate_review_text(review['Place_Rating'])}_")
            else:
                st.info("Belum ada ulasan untuk tempat ini.")
            
            st.markdown("---")
    else:
        st.warning("Tempat wisata tidak ditemukan. Coba kata kunci lain.")

# ==== REKOMENDASI BERDASARKAN USER ====
st.title("🏞️ Rekomendasi Wisata Magelang")
st.markdown("Sistem Rekomendasi Menggunakan *Collaborative Filtering*")
st.markdown("---")

rekomendasi = recommend_places(selected_user, top_n=3)

st.header(f"✨ Rekomendasi Teratas untuk Anda (User {selected_user})")

if rekomendasi:
    cols = st.columns(3)
    for i, (place_info, skor) in enumerate(rekomendasi):
        with cols[i]:
            if 'Image_URL' in place_info and pd.notna(place_info['Image_URL']):
                st.image(place_info['Image_URL'], caption=f"Kategori: {place_info.get('Category', 'N/A')}")
            else:
                st.image("https://via.placeholder.com/400x300.png?text=Gambar+Tidak+Tersedia")
            st.subheader(place_info['Place_Name'])
            st.markdown(f"**🌟 Prediksi Rating: {skor:.2f} / 5.0**")
            st.write(place_info.get('Description', 'Deskripsi tidak tersedia.'))
else:
    st.warning("User ini telah menilai semua tempat wisata. Tidak ada rekomendasi baru.")

st.markdown("---")

# ==== RIWAYAT RATING ====
st.header(f"⭐ Riwayat Rating Anda")
rated_history = get_user_rated_history(selected_user)
if rated_history:
    with st.expander("Lihat tempat yang sudah Anda nilai"):
        for nama, rating in rated_history:
            st.markdown(f"* **{nama}** — `{int(rating)} ⭐`")
else:
    st.info(f"User {selected_user} belum memberikan rating pada tempat manapun.")
