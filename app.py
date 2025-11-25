import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Rekomendasi Wisata Magelang",
    page_icon="🏞️",
    layout="wide"
)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    rating_df = pd.read_csv("Dataset_Rating_Mgl.csv")
    place_df = pd.read_csv("Dataset_tourisMagelang.csv")
    user_df = pd.read_csv("Dataset_usermgl.csv")
    return rating_df, place_df, user_df

rating_df, place_df, user_df = load_data()

# ===============================
# SEARCH WISATA (Dengan Sorting)
# ===============================
def search_place(keyword):
    keyword_lower = keyword.lower()

    # Cari berdasarkan nama & deskripsi
    name_match = place_df[place_df['Place_Name'].str.contains(keyword, case=False, na=False)]
    desc_match = place_df[place_df['Description'].str.contains(keyword, case=False, na=False)]

    # Gabungkan
    results = pd.concat([name_match, desc_match]).drop_duplicates()

    if results.empty:
        return results

    # Tambahkan kolom rata-rata rating
    results["avg_rating"] = results["Place_Name"].apply(
        lambda x: rating_df[rating_df["Place_Name"] == x]["Place_Rating"].mean()
    )

    # Urutkan dari rating tertinggi → terendah
    results = results.sort_values("avg_rating", ascending=False)

    return results

# ===============================
# UI STREAMLIT
# ===============================
st.title("🏞️ Sistem Rekomendasi Wisata Magelang")
st.caption("Menggunakan Matrix Factorization (Model Manual) — *Tanpa User ID*")

st.markdown("---")

# ===============================
# SEARCH BAR
# ===============================
search_query = st.text_input("🔍 Cari Tempat Wisata...", placeholder="Misal: Borobudur")

if search_query:
    results = search_place(search_query)

    if results.empty:
        st.warning("❌ Tempat tidak ditemukan.")
    else:
        for _, row in results.iterrows():
            st.subheader(f"📍 {row['Place_Name']}")

            # Highlight keyword
            desc = row["Description"]
            highlighted = re.sub(f"(?i)({search_query})", r"**\1**", desc)
            st.markdown(highlighted)

            # Rating rata-rata
            avg = row["avg_rating"]
            st.write(f"⭐ **Rata-rata Rating: {avg:.2f}/5.0**")

            st.markdown("---")

else:
    st.info("Masukkan nama tempat wisata untuk mencari informasi.")
