import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# 🔧 KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Rekomendasi Wisata Magelang",
    page_icon="🏞️",
    layout="wide"
)

# ===============================
# 📂 LOAD DATA & MODEL
# ===============================
@st.cache_data
def load_data():
    try:
        rating_df = pd.read_csv("Dataset_Rating_Mgl.csv")
        place_df = pd.read_csv("Dataset_tourisMagelang.csv")
        user_df = pd.read_csv("Dataset_usermgl.csv")
    except FileNotFoundError:
        st.error("❌ File dataset tidak ditemukan. Pastikan semua CSV sudah diunggah.")
        return None, None, None
    return rating_df, place_df, user_df

@st.cache_data
def load_model():
    try:
        rating_matrix = joblib.load("rating_matrix.pkl")
        user_similarity_df = joblib.load("user_similarity.pkl")
    except FileNotFoundError:
        st.error("❌ File model (.pkl) tidak ditemukan.")
        return None, None
    return rating_matrix, user_similarity_df

rating_df, place_df, user_df = load_data()
rating_matrix, user_similarity_df = load_model()

if rating_df is None or rating_matrix is None:
    st.stop()

# ===============================
# 🧠 FUNGSI REKOMENDASI
# ===============================
def predict_rating(user_id, place_id):
    """Prediksi rating untuk user tertentu terhadap tempat tertentu."""
    if place_id in rating_matrix.columns:
        sim_scores = user_similarity_df.loc[user_id]
        ratings = rating_matrix[place_id]
        mask = ratings.notna() & (ratings.index != user_id)
        relevant_sims = sim_scores[mask]
        relevant_ratings = ratings[mask]
        if relevant_sims.sum() > 0:
            return np.dot(relevant_sims, relevant_ratings) / relevant_sims.sum()
    return None

# ===============================
# 🔍 FUNGSI SEARCH TEMPAT
# ===============================
def search_place(place_name):
    result = place_df[place_df['Place_Name'].str.contains(place_name, case=False, na=False)]
    return result

# ===============================
# 💬 FUNGSI MENAMPILKAN ULASAN
# ===============================
def get_reviews_for_place(place_name):
    """Ambil ulasan pengguna lain dari tempat tertentu."""
    reviews = rating_df.merge(user_df, on="User_Id", how="left")
    reviews = reviews[reviews['Place_Name'].str.lower() == place_name.lower()]
    return reviews

# ===============================
# 🖥️ ANTARMUKA STREAMLIT
# ===============================
st.title("🏞️ Sistem Rekomendasi Wisata Magelang")
st.caption("Menggunakan *Collaborative Filtering* dengan algoritma *Matrix Factorization*")

st.markdown("---")

# 🔍 Input Search Tempat Wisata
search_query = st.text_input("🔍 Cari Tempat Wisata", placeholder="Misal: Borobudur atau Gunung Tidar")

if search_query:
    results = search_place(search_query)

    if results.empty:
        st.warning("Tempat tidak ditemukan. Coba kata lain.")
    else:
        for idx, row in results.iterrows():
            st.subheader(f"📍 {row['Place_Name']}")
            
            # Deskripsi
            if 'Description' in row and not pd.isna(row['Description']):
                st.write(f"📝 {row['Description']}")
            else:
                st.info("Belum ada deskripsi untuk tempat ini.")
            
            # Rating rata-rata
            avg_rating = rating_df[rating_df['Place_Name'].str.lower() == row['Place_Name'].lower()]['Place_Rating'].mean()
            st.write(f"⭐ **Rata-rata Rating:** {avg_rating:.2f}/5.0")
            
            # Ulasan pengguna lain
            reviews = get_reviews_for_place(row['Place_Name'])
            if not reviews.empty:
                st.write("💬 **Ulasan Pengguna:**")
                for _, review in reviews.iterrows():
                    user_info = f"{review['Gender']}, {review['Age']} ({review['Regional']})"
                    st.markdown(f"- 🧍‍♂️ **{user_info}** memberi rating `{int(review['Place_Rating'])}` ⭐")
            else:
                st.info("Belum ada ulasan pengguna untuk tempat ini.")
            
            st.markdown("---")

else:
    st.info("Masukkan nama tempat wisata untuk melihat deskripsi, rating, dan ulasan pengguna lain.")

st.sidebar.success("✅ Sistem siap digunakan!")
st.sidebar.caption("Dibuat oleh Armis Dayanti ❤️")
