import streamlit as st
import pandas as pd
import numpy as np
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
# SEARCH FUNCTION
# ===============================
def search_place(keyword, selected_category=None):
    df = place_df.copy()

    # Filter kategori jika dipilih
    if selected_category != "Semua Kategori":
        df = df[df["Category"] == selected_category]

    # Jika tidak ada keyword → tampilkan semua tempat sesuai kategori
    if keyword == "":
        return df

    keyword_lower = keyword.lower()

    name_match = df[df["Place_Name"].str.contains(keyword, case=False, na=False)]
    desc_match = df[df["Description"].str.contains(keyword, case=False, na=False)]

    results = pd.concat([name_match, desc_match]).drop_duplicates()
    return results

# ===============================
# UI STREAMLIT
# ===============================
st.title("🏞️ Sistem Rekomendasi Wisata Magelang")
st.caption("Menggunakan Matrix Factorization (Model Manual) — *Tanpa User ID*")

st.markdown("---")

# ===============================
# FILTER KATEGORI
# ===============================
all_categories = ["Semua Kategori"] + sorted(place_df["Category"].unique())

selected_category = st.selectbox("🎯 Filter Kategori", all_categories)

# ===============================
# SEARCH BAR
# ===============================
search_query = st.text_input("🔍 Cari Tempat Wisata...", placeholder="Misal: Borobudur, gunung, museum...")

# Ambil hasil
results = search_place(search_query, selected_category)

# ===============================
# TAMPILKAN HASIL
# ===============================
if results.empty:
    st.warning("❌ Tidak ditemukan tempat wisata sesuai pencarian.")
else:
    # Urutkan berdasarkan rating tertinggi (Rating_Gmaps)
    results = results.sort_values(by="Rating_Gmaps", ascending=False)

    for _, row in results.iterrows():
        st.subheader(f"📍 {row['Place_Name']}")

        # Highlight text
        desc = row["Description"]
        if search_query:
            desc = re.sub(f"(?i)({search_query})", r"**\1**", desc)

        st.markdown(desc)

        # Rating
        st.write(f"⭐ Rating Google Maps: **{row['Rating_Gmaps']}** / 5.0")

        st.write(f"🏷️ Kategori: **{row['Category']}**")
        st.write("---")
