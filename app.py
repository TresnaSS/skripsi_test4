import streamlit as st
import pickle
import pandas as pd
from deep_translator import GoogleTranslator
import re

# Load model dan vectorizer
with open("svc.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load data untuk rekomendasi
df = pd.read_csv("data_film_clean.csv")

# Function translate
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        return text

# Function preprocessing
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text.lower()

# Show instructions on first load
if "show_guide" not in st.session_state:
    st.session_state.show_guide = True

if st.session_state.show_guide:
    with st.expander("📘 Panduan Penggunaan", expanded=True):
        st.markdown("""
        **Langkah-langkah Penggunaan Aplikasi:**
        1. Masukkan sinopsis film yang ingin diprediksi.
        2. Jika sinopsis menggunakan bahasa asing, klik tombol **Translate ke Bahasa Inggris** terlebih dahulu.
        3. Setelah itu, klik tombol **Prediksi Genre** untuk melihat hasil prediksi genre.
        4. Aplikasi akan menampilkan rekomendasi film berdasarkan genre tersebut.
        """)
        if st.button("Tutup Panduan"):
            st.session_state.show_guide = False

# Judul aplikasi
st.title("🎬 Prediksi Genre Film Berdasarkan Sinopsis")

# Input teks
input_text = st.text_area("Masukkan sinopsis film (minimal 20 kata):")

# Reset hasil translate jika input berubah
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

if input_text != st.session_state.last_input:
    st.session_state.translated_text = ""
    st.session_state.last_input = input_text

# Inisialisasi state untuk teks terjemahan
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""

# Tombol Translate
if st.button("🔁 Translate ke Bahasa Inggris"):
    st.session_state.translated_text = translate_to_english(input_text)
    st.success("Hasil Translate:")
    st.write(st.session_state.translated_text)

# Tombol Prediksi
if st.button("🎯 Prediksi Genre"):
    text_to_use = st.session_state.translated_text or input_text
    if len(text_to_use.split()) < 20:
        st.warning("❗ Sinopsis terlalu pendek. Masukkan minimal 20 kata.")
    else:
        cleaned = clean_text(text_to_use)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"Genre yang diprediksi: **{prediction}**")

        # Rekomendasi film
        st.subheader("📚 Rekomendasi Film:")
        recommended = df[df['clean_synopsis'].str.contains(prediction.lower(), na=False)]

        if not recommended.empty:
            for Title in recommended['Title'].head(5):
                st.markdown(f"- 🎞️ {Title}")
        else:
            st.warning("Tidak ada rekomendasi film yang ditemukan untuk genre ini.")
