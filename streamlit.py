import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model dan komponen
with open('naive_bayes_model.pkl', 'rb') as file:
    model_full = pickle.load(file)

with open('naive_bayes_chi2_model.pkl', 'rb') as file:
    model_chi2 = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

with open('feature_names.pkl', 'rb') as file:
    selected_features = pickle.load(file)

with open('feature_names_full.pkl', 'rb') as file:
    feature_names_full = pickle.load(file)

# Fungsi preprocessing input pengguna
def preprocess_input(input_data, feature_names):
    # Encoding kategori
    for col, encoder in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = encoder.transform([input_data[col]])[0]
    
    # Skala numerik
    if 'Panjang' in input_data.columns and 'pH' in input_data.columns:
        input_data[['Panjang', 'pH']] = scaler.transform(input_data[['Panjang', 'pH']])
    
    # Urutkan fitur sesuai kebutuhan model
    input_data = input_data[feature_names]
    return input_data

# Judul aplikasi
st.title("Klasifikasi Tanaman Padi dengan Naive Bayes")

# Pilihan metode klasifikasi
method = st.selectbox(
    "Pilih Metode Klasifikasi:",
    ["Naive Bayes (Semua Fitur)", "Naive Bayes dengan Seleksi Fitur (Chi-Square)"]
)

# Input fitur dari pengguna
if method == "Naive Bayes (Semua Fitur)":
    st.header("Input Semua Fitur")
    varietas = st.selectbox("Varietas", ["Pilih...", "Inpari 42", "Beras Merah", "Siam - Siam"])
    panjang = st.number_input("Panjang (cm)", min_value=0.0, max_value=100.0, step=0.1)
    bentuk = st.selectbox("Bentuk", ["Pilih...", "Bulat", "Panjang", "Ramping"])
    warna = st.selectbox("Warna", ["Pilih...", "Putih", "Merah", "Hitam"])
    rasa = st.selectbox("Rasa", ["Pilih...", "Pulen", "Kurang Pulen"])
    teknik = st.selectbox("Teknik", ["Pilih...", "Konvensional", "Jajar Legowo"])
    musim = st.selectbox("Musim", ["Pilih...", "Hujan", "Kemarau"])
    hama = st.selectbox("Hama", ["Pilih...", "Wereng Hijau", "Tikus", "Burung"])
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
    
    # Validasi input
    if "Pilih..." in [varietas, bentuk, warna, rasa, teknik, musim, hama]:
        st.warning("Mohon isi semua pilihan sebelum melakukan prediksi.")
    else:
        input_data = pd.DataFrame({
            "Varietas": [varietas],
            "Panjang": [panjang],
            "Bentuk": [bentuk],
            "Warna": [warna],
            "Rasa": [rasa],
            "Teknik": [teknik],
            "Musim": [musim],
            "Hama": [hama],
            "pH": [ph]
        })

        # Preprocessing input
        input_data = preprocess_input(input_data, feature_names=feature_names_full)

        # Prediksi
        if st.button("Prediksi"):
            prediction = model_full.predict(input_data)
            if prediction[0] == 0:
                st.success("Kualitas Tanaman Padi: Bagus")
            else:
                st.error("Kualitas Tanaman Padi: Kurang Bagus")

elif method == "Naive Bayes dengan Seleksi Fitur (Chi-Square)":
    st.header("Input Fitur Terpilih")
    input_fields = {}
    for feature in selected_features:
        if feature == "Varietas":
            input_fields[feature] = st.selectbox("Varietas", ["Pilih...", "Inpari 42", "Beras Merah", "Siam - Siam"])
        elif feature == "Bentuk":
            input_fields[feature] = st.selectbox("Bentuk", ["Pilih...", "Bulat", "Panjang", "Ramping"])
        elif feature == "Warna":
            input_fields[feature] = st.selectbox("Warna", ["Pilih...", "Putih", "Merah", "Hitam"])
        elif feature == "Rasa":
            input_fields[feature] = st.selectbox("Rasa", ["Pilih...", "Pulen", "Kurang Pulen"])
        elif feature == "Teknik":
            input_fields[feature] = st.selectbox("Teknik", ["Pilih...", "Konvensional", "Jajar Legowo"])
        elif feature == "Musim":
            input_fields[feature] = st.selectbox("Musim", ["Pilih...", "Hujan", "Kemarau"])
        elif feature == "Hama":
            input_fields[feature] = st.selectbox("Hama", ["Pilih...", "Wereng Hijau", "Tikus", "Burung"])
        elif feature == "pH":
            input_fields[feature] = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)

    if "Pilih..." in input_fields.values():
        st.warning("Mohon isi semua pilihan sebelum melakukan prediksi.")
    else:
        input_data = pd.DataFrame(input_fields, index=[0])

        # Preprocessing input
        input_data = preprocess_input(input_data, feature_names=selected_features)

        # Prediksi
        if st.button("Prediksi"):
            prediction = model_chi2.predict(input_data)
            if prediction[0] == 0:
                st.success("Kualitas Tanaman Padi: Bagus")
            else:
                st.error("Kualitas Tanaman Padi: Kurang Bagus")
