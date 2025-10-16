import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === 1️⃣ Setup Awal ===
DATA_PATH = "data.xlsx"
REPORT_FILE = "training_report.txt"

start_time = time.time()
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("📘 Memulai pelatihan model Naive Bayes...")
print(f"🕒 Waktu mulai: {timestamp}\n")

# === 2️⃣ Load Dataset ===
df = pd.read_excel(DATA_PATH)
TARGET = "Kualitas"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("✅ Dataset dimuat:", df.shape)
print("Kolom:", list(df.columns))

# === 3️⃣ Encoding kategori ===
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    label_encoders[col] = encoder

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
label_encoders[TARGET] = target_encoder

# === 4️⃣ Normalisasi numerik ===
scaler = MinMaxScaler()
X_scaled = X.copy()
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
X_scaled[num_cols] = scaler.fit_transform(X[num_cols])

# === 5️⃣ Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# === 6️⃣ Model Full Fitur ===
model_full = GaussianNB()
model_full.fit(X_train, y_train)
pred_full = model_full.predict(X_test)
acc_full = accuracy_score(y_test, pred_full)
report_full = classification_report(y_test, pred_full, target_names=target_encoder.classes_)

# === 7️⃣ Model Chi-Square ===
selector = SelectKBest(score_func=chi2, k=min(5, X_train.shape[1]))
X_train_chi2 = selector.fit_transform(X_train, y_train)
X_test_chi2 = selector.transform(X_test)
selected_features = X.columns[selector.get_support()]

model_chi2 = GaussianNB()
model_chi2.fit(X_train_chi2, y_train)
pred_chi2 = model_chi2.predict(X_test_chi2)
acc_chi2 = accuracy_score(y_test, pred_chi2)
report_chi2 = classification_report(y_test, pred_chi2, target_names=target_encoder.classes_)

# === 8️⃣ Visualisasi ===
def plot_conf_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    labels = target_encoder.classes_
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title(title)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(x=TARGET, data=df, palette="viridis")
plt.title("Distribusi Kelas Target (Kualitas)")
plt.savefig("viz_target_distribution.png", bbox_inches="tight")
plt.close()

plot_conf_matrix(y_test, pred_full, "Confusion Matrix - Semua Fitur", "viz_cm_full.png")
plot_conf_matrix(y_test, pred_chi2, "Confusion Matrix - Chi-Square", "viz_cm_chi2.png")

plt.figure(figsize=(6, 4))
sns.barplot(x=["Semua Fitur", "Chi-Square"], y=[acc_full, acc_chi2], palette="coolwarm")
plt.ylabel("Akurasi")
plt.title("Perbandingan Akurasi Model Naive Bayes")
plt.ylim(0, 1)
plt.savefig("viz_accuracy_comparison.png", bbox_inches="tight")
plt.close()

# === 9️⃣ Simpan Model dan Komponen ===
with open("naive_bayes_model.pkl", "wb") as f:
    pickle.dump(model_full, f)
with open("naive_bayes_chi2_model.pkl", "wb") as f:
    pickle.dump(model_chi2, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("feature_names_full.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)
with open("feature_names.pkl", "wb") as f:
    pickle.dump(list(selected_features), f)

# === 🔟 Logging ke File ===
end_time = time.time()
duration = end_time - start_time

log_text = f"""
=============================
🧠 TRAINING REPORT
=============================
📅 Tanggal: {timestamp}
🕒 Durasi: {duration:.2f} detik
📘 Dataset: {DATA_PATH}
🧩 Jumlah Fitur: {len(X.columns)}
🎯 Target: {TARGET}

📊 Akurasi Naive Bayes (Semua Fitur): {acc_full:.2%}
📊 Akurasi Naive Bayes (Chi-Square): {acc_chi2:.2%}

🔍 Fitur Terpilih (Chi-Square):
{list(selected_features)}

--- Laporan Lengkap Naive Bayes (Semua Fitur) ---
{report_full}

--- Laporan Lengkap Naive Bayes (Chi-Square) ---
{report_chi2}

=============================
"""

with open(REPORT_FILE, "a", encoding="utf-8") as f:
    f.write(log_text + "\n")

print(log_text)
print("✅ Semua file model, visualisasi, dan laporan berhasil disimpan!")
print("📂 File laporan: training_report.txt")
