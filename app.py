import streamlit as st
import pandas as pd
import numpy as np
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
# 📂 LOAD DATASET
# ===============================
@st.cache_data
def load_data():
    rating_df = pd.read_csv("Dataset_Rating_Mgl.csv")
    place_df = pd.read_csv("Dataset_tourisMagelang.csv")
    user_df  = pd.read_csv("Dataset_usermgl.csv")
    return rating_df, place_df, user_df

rating_df, place_df, user_df = load_data()

# ===============================
# 🔢 MEMBANGUN USER–ITEM MATRIX
# ===============================
@st.cache_data
def build_rating_matrix():
    matrix = rating_df.pivot_table(
        index="User_Id",
        columns="Place_Id",
        values="Place_Rating"
    ).fillna(0)

    return matrix

rating_matrix = build_rating_matrix()

# ===============================
# 🔢 MATRIX FACTORIZATION (SVD MANUAL)
# ===============================
@st.cache_data
def train_manual_svd(R, k=10, alpha=0.002, beta=0.02, steps=200):
    """Melakukan matrix factorization manual (gradient descent)."""
    R = np.array(R)
    num_users, num_items = R.shape

    # Inisialisasi faktor laten
    P = np.random.rand(num_users, k)
    Q = np.random.rand(num_items, k)

    # Gradient Descent
    for step in range(steps):
        for u in range(num_users):
            for i in range(num_items):
                if R[u][i] > 0:  # hanya data yang ada rating
                    error = R[u][i] - np.dot(P[u], Q[i])
                    P[u] += alpha * (2 * error * Q[i] - beta * P[u])
                    Q[i] += alpha * (2 * error * P[u] - beta * Q[i])

    return P, Q

P, Q = train_manual_svd(rating_matrix)

# ===============================
# ⭐ PREDIKSI RATING
# ===============================
def predict_rating_manual(user_index, item_index):
    return np.dot(P[user_index], Q[item_index])

# ===============================
# 🎯 FUNGSI REKOMENDASI
# ===============================
def recommend_places(user_id, top_n=5):

    user_index = list(rating_matrix.index).index(user_id)
    predictions = []

    for place_id in rating_matrix.columns:
        item_index = list(rating_matrix.columns).index(place_id)

        # Jika belum dirating user
        if rating_df[(rating_df.User_Id == user_id) & (rating_df.Place_Id == place_id)].empty:
            pred = predict_rating_manual(user_index, item_index)
            place_name = place_df[place_df["Place_Id"] == place_id]["Place_Name"].values[0]
            predictions.append((place_name, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]

# ===============================
# 🔍 SEARCH WISATA
# ===============================
def search_place(keyword):
    keyword_lower = keyword.lower()

    name_match = place_df[place_df['Place_Name'].str.contains(keyword, case=False, na=False)]
    desc_match = place_df[place_df['Description'].str.contains(keyword, case=False, na=False)]

    results = pd.concat([name_match, desc_match]).drop_duplicates()

    if results.empty:
        return results

    def score(row):
        return row["Place_Name"].lower().count(keyword_lower) * 2 + \
               row["Description"].lower().count(keyword_lower)

    results["Relevance"] = results.apply(score, axis=1)
    return results.sort_values("Relevance", ascending=False)

# ===============================
# 🖥️ ANTARMUKA STREAMLIT
# ===============================
st.title("🏞️ Sistem Rekomendasi Wisata Magelang")
st.caption("Menggunakan Matrix Factorization (SVD Manual NumPy)")

st.markdown("---")

# ===============================
# 🔍 SEARCH BAR
# ===============================
search_query = st.text_input("🔍 Cari Tempat Wisata", placeholder="Misal: Borobudur")

if search_query:
    results = search_place(search_query)

    if results.empty:
        st.warning("Tidak ditemukan.")
    else:
        for _, row in results.iterrows():
            st.subheader(f"📍 {row['Place_Name']}")
            highlighted_desc = re.sub(f"(?i)({search_query})", r"**\1**", row["Description"])
            st.markdown("📝 " + highlighted_desc)
            avg = rating_df[rating_df["Place_Name"] == row["Place_Name"]]["Place_Rating"].mean()
            st.write(f"⭐ Rata-rata Rating: {avg:.2f}")

            st.markdown("---")

# ===============================
# 🎯 REKOMENDASI USER
# ===============================
st.subheader("🎯 Rekomendasi Berdasarkan User ID")

selected_user = st.selectbox(
    "Pilih User ID:",
    sorted(user_df["User_Id"].unique())
)

if st.button("Tampilkan Rekomendasi"):
    rekom = recommend_places(selected_user)

    for place, score in rekom:
        st.markdown(f"- **{place}** — Prediksi Rating: `{score:.2f}` ⭐")

st.sidebar.caption("Sistem Rekomendasi — SVD Manual NumPy")
