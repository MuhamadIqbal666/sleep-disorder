import streamlit as st
import matplotlib.image as mpimg

def app():
    st.title("Aplikasi Prediksi Sleep Disorder")

    # Memuat gambar
    image_path = 'sleep.jpg'
    image = mpimg.imread(image_path)

    st.text("Welcome to aplikasi prediksi penyakit Sleep Disorder!")
    st.image(image, width=300)  

# Jalankan aplikasi
if __name__ == '__main__':
    app()
