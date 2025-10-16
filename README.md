# 🌾 Klasifikasi Tanaman Padi Menggunakan Naive Bayes dan Seleksi Fitur Chi-Square

Proyek ini mengimplementasikan **model Machine Learning** untuk mengklasifikasikan kualitas tanaman padi berdasarkan karakteristik agronomis menggunakan **algoritma Naive Bayes** dan **seleksi fitur Chi-Square**.  
Proyek ini dilengkapi dengan **aplikasi web interaktif berbasis Streamlit**, skrip otomatis untuk pelatihan model, visualisasi performa, serta lingkungan pengembangan yang dapat direproduksi (Dev Container) untuk VS Code dan Docker.

---

## 🚀 Fitur Utama

✅ Klasifikasi kualitas tanaman padi menggunakan Naive Bayes  
✅ Seleksi fitur otomatis dengan metode Chi-Square  
✅ Antarmuka web interaktif berbasis Streamlit  
✅ Skrip otomatis pelatihan model (`train_model.py`)  
✅ Prediksi real-time dengan preprocessing otomatis  
✅ Visualisasi performa pelatihan dan confusion matrix  
✅ Dukungan Dev Container (VS Code & Docker)  
✅ Logging hasil pelatihan otomatis (`training_report.txt`)

---

## 🧠 Ringkasan Model

| Jenis Model | Deskripsi | Akurasi |
|--------------|------------|----------|
| Naive Bayes (Semua Fitur) | Menggunakan seluruh fitur yang tersedia | ~94–96% |
| Naive Bayes + Chi-Square | Menggunakan 5 fitur paling relevan hasil seleksi Chi-Square | ~96–98% |

> (*Nilai akurasi dapat sedikit berbeda tergantung pada pembagian data.*)

---

## 📊 Deskripsi Dataset

**File:** `data.xlsx`

| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| Varietas | Kategorikal | Jenis varietas padi (mis. Inpari 42, Beras Merah) |
| Panjang | Numerik | Panjang butir padi (cm) |
| Bentuk | Kategorikal | Bentuk butir padi |
| Warna | Kategorikal | Warna butir padi |
| Rasa | Kategorikal | Cita rasa nasi hasil tanak |
| Teknik | Kategorikal | Metode budidaya |
| Musim | Kategorikal | Musim tanam |
| Hama | Kategorikal | Jenis hama dominan |
| pH | Numerik | Tingkat keasaman tanah |
| **Kualitas** | Target | Kualitas tanaman padi (Bagus / Kurang Bagus) |

---

## 🧮 Input Data

| Kolom        | Deskripsi                                                               |
| ------------ | ----------------------------------------------------------------------- |
| Varietas     | Jenis varietas padi (contoh: *Inpari 42*, *Beras Merah*, *Siam - Siam*) |
| Panjang      | Panjang butir padi dalam cm (contoh: 9.2)                               |
| Bentuk       | Bentuk butir padi (contoh: *Bulat*, *Ramping*, *Panjang*)               |
| Warna        | Warna beras (contoh: *Putih*, *Merah*, *Hitam*)                         |
| Rasa         | Karakteristik rasa (contoh: *Pulen*, *Kurang Pulen*)                    |
| Teknik       | Metode penanaman (contoh: *Jajar Legowo*, *Konvensional*)               |
| Musim        | Musim tanam (contoh: *Hujan*, *Kemarau*)                                |
| Hama         | Jenis hama yang menyerang (contoh: *Tikus*, *Wereng Hijau*, *Burung*)   |
| pH           | Tingkat keasaman tanah (0.0–14.0)                                       |
| **Kualitas** | Label target (*Bagus* / *Kurang Bagus*)                                 |


---

## 🧩 Struktur Proyek

📦 Klasifikasi-Tanaman-Padi
├── streamlit.py
├── train_model.py
├── data.xlsx
├── naive_bayes_model.pkl
├── naive_bayes_chi2_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── feature_names.pkl
├── feature_names_full.pkl
├── training_report.txt
├── requirements.txt
├── LICENSE
├── .gitignore
└── .devcontainer/
└── devcontainer.json


---

## 🖥️ Aplikasi Streamlit

### ▶️ Jalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run streamlit.py
```
Pengguna akan diminta untuk mengisi form interaktif di halaman web, seperti:

Varietas: Pilih jenis padi dari dropdown

Panjang: Masukkan nilai numerik (cm)

Bentuk, Warna, Rasa, Teknik, Musim, Hama: Pilih dari opsi yang tersedia

pH: Masukkan nilai antara 0.0–14.0

Setelah semua data diisi, tekan tombol "Prediksi" untuk melihat hasilnya:

🌱 “Kualitas Tanaman Padi: Bagus”
atau
❌ “Kualitas Tanaman Padi: Kurang Bagus”

---

### 🌐 Deploy ke Streamlit Cloud

Push repository ke GitHub

Buka Streamlit Cloud

Pilih repository dan tentukan streamlit.py sebagai entry point

Tunggu hingga proses build dan aplikasi berjalan 🚀

Aplikasi dapat diakses melalui: http://localhost:8501

---

## ⚙️ Pelatihan Model Otomatis

Kamu dapat melatih ulang model kapan pun dengan menjalankan: python train_model.py

Skrip ini akan:

Melatih dua model: Naive Bayes (semua fitur) dan Naive Bayes + Chi-Square

Menyimpan file .pkl terbaru

Menghasilkan visualisasi:

      viz_target_distribution.png

      viz_cm_full.png

      viz_cm_chi2.png

      viz_accuracy_comparison.png

Menambahkan laporan hasil pelatihan ke file training_report.txt

---

## 📊 Contoh Visualisasi
🔹 Distribusi Kelas Target

🔹 Confusion Matrix – Semua Fitur

🔹 Confusion Matrix – Chi-Square

🔹 Perbandingan Akurasi Model

---

## 🧾 Contoh Hasil Laporan Pelatihan

Cuplikan dari training_report.txt:
=============================
🧠 TRAINING REPORT
=============================
📅 Tanggal: 2025-10-16 14:35:22
🕒 Durasi: 2.84 detik
📘 Dataset: data.xlsx
🧩 Jumlah Fitur: 9
🎯 Target: Kualitas

📊 Akurasi Naive Bayes (Semua Fitur): 94.23%
📊 Akurasi Naive Bayes (Chi-Square): 96.15%

🔍 Fitur Terpilih (Chi-Square):
['Panjang', 'Rasa', 'Teknik', 'Musim', 'pH']

---

## 📦 Dependencies

numpy==1.26.4
pandas==2.2.2
scikit-learn==1.6.0
imbalanced-learn==0.14.0
streamlit==1.37.0
matplotlib==3.9.2
seaborn==0.13.2
joblib==1.4.2
protobuf==4.25.3
altair<5

---

## 🌟 Ucapan Terima Kasih

Proyek ini dikembangkan sebagai bagian dari penelitian penerapan Machine Learning dalam bidang pertanian, khususnya untuk mendukung pengambilan keputusan dalam meningkatkan kualitas dan produktivitas tanaman padi.

💡 "Mengubah data menjadi wawasan, dan wawasan menjadi pertumbuhan."
— Toba Jordi Naibaho