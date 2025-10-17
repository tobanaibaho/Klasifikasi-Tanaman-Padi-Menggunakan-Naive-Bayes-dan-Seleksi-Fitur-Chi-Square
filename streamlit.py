# ======================================================
# Streamlit App Setup - by Toba Jordi Naibaho
# ======================================================

import streamlit as st

# Konfigurasi dasar halaman (harus di baris awal sebelum elemen UI lain)
st.set_page_config(
    page_title="Klasifikasi Tanaman Padi 🌾",
    page_icon="🌾",                     # Bisa diganti emoji atau path ke file .png
    layout="wide",                      # Gunakan layout lebar (lebih profesional)
    initial_sidebar_state="expanded"    # Sidebar terbuka saat pertama kali
)

# Optional: custom CSS tambahan (agar tampil elegan)
st.markdown("""
    <style>
    /* Gaya teks judul */
    h1 {
        color: #228B22;
        font-weight: 700;
        text-align: center;
    }
    /* Tombol Streamlit */
    div.stButton > button {
        background-color: #228B22;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-size: 1em;
        font-weight: 600;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #1a6b1a;
        color: #ffffff;
        transform: scale(1.03);
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# Dashboard Pembuka - Klasifikasi Tanaman Padi 🌾
# ======================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar navigasi
menu = st.sidebar.radio(
    "Navigasi Aplikasi",
    ["🏠 Dashboard", "🔍 Prediksi Kualitas Padi"]
)

# ======================================================
# 1️⃣ DASHBOARD UTAMA
# ======================================================
if menu == "🏠 Dashboard":
    st.title("🌾 Dashboard Klasifikasi Tanaman Padi")
    st.markdown("""
    Selamat datang di **Aplikasi Klasifikasi Tanaman Padi**.  
    Aplikasi ini menggunakan algoritma **Naive Bayes** dengan **Seleksi Fitur Chi-Square**  
    untuk memprediksi kualitas padi berdasarkan parameter morfologi dan lingkungan.

    ---
    """)

    # Membaca dataset (pastikan file 'data.xlsx' ada di root)
    try:
        df = pd.read_excel("data.xlsx")

        st.subheader("📊 Ringkasan Dataset")
        st.dataframe(df.head())

        # Menampilkan info dataset
        st.markdown(f"""
        - Jumlah data: **{df.shape[0]} baris**
        - Jumlah fitur: **{df.shape[1]} kolom**
        - Kolom: `{', '.join(df.columns)}`
        """)

        # ======================================================
        # Visualisasi sederhana
        st.subheader("📈 Distribusi Kualitas Tanaman")
        if "Kualitas" in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(x="Kualitas", data=df, palette="Greens", ax=ax)
            ax.set_title("Distribusi Kualitas (Bagus vs Kurang Bagus)")
            st.pyplot(fig)
        else:
            st.info("Kolom 'Kualitas' tidak ditemukan di dataset.")

        # ======================================================
        st.subheader("🌾 Statistik Fitur Numerik")
        st.write(df.describe())

        st.markdown("""
        ---
        **Catatan:**  
        Model telah dilatih dengan Naive Bayes dan seleksi fitur Chi-Square  
        untuk meningkatkan akurasi dan mengurangi overfitting.
        """)
        # ======================================================
        # EVALUASI PERFORMA MODEL (dengan preprocessing otomatis)
        # ======================================================
        import pickle
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.subheader("🧠 Evaluasi Performa Model")

        try:
            # Muat semua komponen model
            with open("naive_bayes_model.pkl", "rb") as f:
                model_full = pickle.load(f)
            with open("naive_bayes_chi2_model.pkl", "rb") as f:
                model_chi2 = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open("label_encoders.pkl", "rb") as f:
                label_encoders = pickle.load(f)
            with open("feature_names.pkl", "rb") as f:
                selected_features = pickle.load(f)
            with open("feature_names_full.pkl", "rb") as f:
                feature_names_full = pickle.load(f)

            # Fungsi preprocessing (mirip dengan fungsi di streamlit.py utama)
            def preprocess_input_eval(df, feature_names):
                df_copy = df.copy()
                for col, encoder in label_encoders.items():
                    if col in df_copy.columns:
                        try:
                            df_copy[col] = encoder.transform(df_copy[col])
                        except:
                            df_copy[col] = df_copy[col].map(lambda x: encoder.classes_[0] if x not in encoder.classes_ else x)
                            df_copy[col] = encoder.transform(df_copy[col])
                if "Panjang" in df_copy.columns and "pH" in df_copy.columns:
                    df_copy[["Panjang", "pH"]] = scaler.transform(df_copy[["Panjang", "pH"]])
                return df_copy[feature_names]

            # Jika dataset punya kolom target
            if "Kualitas" in df.columns:
                X_raw = df.drop(columns=["Kualitas"])
                y_true = df["Kualitas"].replace({"Bagus": 0, "Kurang Bagus": 1}).values

                # Preprocessing sebelum evaluasi
                X_full = preprocess_input_eval(X_raw, feature_names_full)
                X_chi2 = preprocess_input_eval(X_raw, selected_features)

                # Prediksi dan evaluasi
                y_pred_full = model_full.predict(X_full)
                y_pred_chi2 = model_chi2.predict(X_chi2)

                metrics = pd.DataFrame({
                    "Model": ["Naive Bayes (Full Features)", "Naive Bayes + Chi-Square"],
                    "Akurasi": [
                        accuracy_score(y_true, y_pred_full),
                        accuracy_score(y_true, y_pred_chi2)
                    ],
                    "Presisi": [
                        precision_score(y_true, y_pred_full, zero_division=0),
                        precision_score(y_true, y_pred_chi2, zero_division=0)
                    ],
                    "Recall": [
                        recall_score(y_true, y_pred_full, zero_division=0),
                        recall_score(y_true, y_pred_chi2, zero_division=0)
                    ],
                    "F1-Score": [
                        f1_score(y_true, y_pred_full, zero_division=0),
                        f1_score(y_true, y_pred_chi2, zero_division=0)
                    ]
                })

                st.dataframe(metrics.style.format({
                    "Akurasi": "{:.2%}",
                    "Presisi": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-Score": "{:.2%}"
                }))

                # Visualisasi perbandingan metrik
                st.subheader("📊 Perbandingan Performa Model")
                fig, ax = plt.subplots(figsize=(7, 4))
                metrics_melted = metrics.melt(id_vars="Model", var_name="Metrik", value_name="Nilai")
                sns.barplot(data=metrics_melted, x="Metrik", y="Nilai", hue="Model", palette="Greens", ax=ax)
                ax.set_title("Perbandingan Akurasi dan Metrik Model")
                ax.set_ylabel("Nilai (0 - 1)")
                st.pyplot(fig)

            else:
                st.info("Kolom target 'Kualitas' tidak ditemukan di dataset. Tidak bisa menghitung akurasi model.")

        except FileNotFoundError:
            st.error("File model atau komponen `.pkl` tidak ditemukan.")
        except Exception as e:
            st.warning("⚠️ Gagal menjalankan evaluasi otomatis.")
            st.caption(str(e))
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        st.info("Pastikan file `data.xlsx` tersedia di direktori proyek Anda.")

# ======================================================
# 2️⃣ MENU PREDIKSI
# ======================================================
elif menu == "🔍 Prediksi Kualitas Padi":
    st.title("🔍 Prediksi Kualitas Tanaman Padi")
    st.markdown("Silakan masukkan parameter tanaman di bawah ini untuk melakukan prediksi kualitas.")
    st.divider()
    # (Bagian form input prediksi yang sudah kamu punya tetap dilanjutkan di sini)


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
