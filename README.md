# AquaCheck — Klasifikasi Kualitas Air Minum

Aplikasi web berbasis **Flask** yang menggunakan **Algoritma Random Forest**
untuk memprediksi apakah air layak diminum berdasarkan 9 parameter fisikokimia.

---

## 🎯 Fitur Utama

- Prediksi real-time kualitas air (Layak / Tidak Layak)
- Analisis probabilitas per kelas
- Pengecekan 9 parameter air secara individual
- Visualisasi Feature Importance
- Akurasi model **99.54%**

---

## 📊 Parameter yang Dianalisis

| Parameter | Satuan | Ambang Batas Aman |
|---|---|---|
| pH Air | — | 6.5 – 8.5 |
| Hardness (Kekerasan) | mg/L | < 200 |
| Total Dissolved Solids | ppm | < 20000 |
| Chloramines (Kloramin) | ppm | < 8 |
| Sulfate (Sulfat) | mg/L | < 350 |
| Conductivity (Konduktivitas) | μS/cm | < 500 |
| Organic Carbon | ppm | < 15 |
| Trihalomethanes | μg/L | < 80 |
| Turbidity (Kekeruhan) | NTU | < 4 |

---

## 🤖 Detail Model

| Parameter | Nilai |
|---|---|
| Algoritma | Random Forest Classifier |
| Jumlah Pohon (n_estimators) | 150 |
| Max Depth | 12 |
| Akurasi Test Set | 99.54% |
| Library | scikit-learn |

---

## 🚀 Cara Menjalankan Lokal

```bash
# 1. Clone repository
git clone https://github.com/username/aquacheck.git
cd aquacheck

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model (hanya pertama kali)
python model/train_model.py

# 4. Jalankan Flask
python app.py
```

Buka browser: **http://localhost:5000**

---

## ☁️ Deployment ke Heroku

```bash
heroku create aquacheck-app
git add .
git commit -m "Initial commit"
git push heroku main
```

---

## 📁 Struktur Proyek

```
water_quality_app/
├── app.py                  ← Flask main application
├── requirements.txt        ← Python dependencies
├── Procfile                ← Heroku config
├── model/
│   ├── train_model.py      ← Script training model
│   ├── rf_model.pkl        ← Trained Random Forest model
│   ├── scaler.pkl          ← StandardScaler
│   └── feature_names.pkl   ← Feature list
├── templates/
│   └── index.html          ← UI (Flask Jinja2)
└── static/
    └── ...
```

---

## 📚 Referensi

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
- Scikit-learn Documentation: https://scikit-learn.org
- Water Potability Dataset: https://www.kaggle.com/datasets/adityakadiwal/water-potability

---

*Tugas 8 — Kecerdasan Buatan · Teknik Informatika · Universitas Bale Bandung*
