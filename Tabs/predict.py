import streamlit as st
from web_functions import predict

def app(df, x, y):

    st.title("Halaman Prediksi Sleep Disorder")
    
    # Jenis Kelamin
    Gender = st.selectbox('Pilih jenis kelamin (Male=0 Female=1)', ['0', '1'])
    Age = st.number_input('Masukkan Umur', 0, 100)
    st.image('occupation.png', caption='Deskripsi pekerjaan', use_column_width=300)
    Occupation = st.selectbox('Pilih Nomor Pekerjaan', ['0','1','2','3','4','5','6','7','8','9','10'])
    Sleep_Duration = st.number_input('Durasi Tidur perhari', 5.9, 8.5)
    Quality_of_Sleep = st.selectbox('Pilih Nilai Kualitas Tidur', ['1','2','3','4','5','6','7''8'])
    Physical_Activity_Level = st.number_input('Tingkat aktivitas fisik (menit perhari)', 0)
    Stress_Level = st.slider('Tingkat stress (Stress ringan-stress berat)', 1, 10)
    BMI_Category = st.selectbox('Pilih BMI Category (0=Normal,1=Obese,2=Overweight)', ['0', '1', '2'])
    Blood_Pressure = st.slider('Pilih tungkat tekanan darah (Rendah-Tinggi)', 0, 23)
    Heart_Rate = st.number_input('Detak jantung (detak permenit)', 0)
    Daily_Steps = st.number_input('Jumlah langkah harian', 0)
    features = [Gender, Age, Occupation, Sleep_Duration, Quality_of_Sleep, Physical_Activity_Level, Stress_Level, BMI_Category, Blood_Pressure, Heart_Rate, Daily_Steps]

    if st.button("Prediksi"):
        prediction, score = predict(x, y, features)
        st.info("Prediksi Sukses...")

        if prediction == 1:
            st.warning('Pasien Mengalami Insomnia')
        elif prediction == 2:
            st.warning('Pasien Mengalami Sleep Apnea')
        else:
            st.success('Pasien Tidak Mengalami Sleep Disorder')

        st.write("Model yang digunakan memiliki tingkat akurasi ", round(score * 100, 2), "%")

