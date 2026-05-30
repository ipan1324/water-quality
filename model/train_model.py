"""
Train Random Forest - Klasifikasi Kualitas Air Minum
Step-by-step interaktif dengan tampilan terminal berwarna
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

G  = "\033[92m"; Y  = "\033[93m"; R  = "\033[91m"; C  = "\033[96m"
W  = "\033[97m"; B  = "\033[94m"; M  = "\033[95m"; DIM= "\033[2m"; RST= "\033[0m"

def line():  print(DIM + "-"*60 + RST)
def pause(): input(Y + "\n[Tekan ENTER untuk lanjut...]" + RST)
def header(step, title, icon="🔷"):
    print("\n" + "="*60)
    print(f"{icon} {C}STEP {step}{RST} — {W}{title}{RST}")
    print("="*60)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "water_potability.csv")

# ── STEP 1 ──────────────────────────────────────────────────────
header(1, "Load Dataset", "📁")
df = pd.read_csv(CSV_PATH)
print(f"  {G}✅ Dataset berhasil dimuat!{RST}")
print(f"  {W}Jumlah data  :{RST} {G}{len(df)} baris{RST}")
print(f"  {W}Jumlah fitur :{RST} {G}{len(df.columns)-1} kolom (+ 1 label){RST}")
line(); pause()

# ── STEP 2 ──────────────────────────────────────────────────────
header(2, "Cek Data", "🔍")
print(f"  {W}Kolom dataset:{RST}")
for col in df.columns:
    print(f"    {DIM}•{RST} {col}")
print()
for col, cnt in df.isnull().sum().items():
    if cnt > 0:
        print(f"  {Y}⚠  Missing — {col}: {cnt}{RST}")
if df.isnull().sum().sum() == 0:
    print(f"  {G}Missing values: 0 (tidak ada masalah){RST}")
line(); pause()

# ── STEP 3 ──────────────────────────────────────────────────────
header(3, "Preprocessing", "⚙️")
for col in df.columns[df.isnull().any()]:
    for cls in df["Potability"].unique():
        med = df.loc[df["Potability"] == cls, col].median()
        df.loc[(df["Potability"] == cls) & (df[col].isnull()), col] = med
X = df.drop("Potability", axis=1)
y = df["Potability"].astype(int)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  {G}✅ Fitur dan label dipisahkan{RST}")
print(f"  {G}✅ Missing values diisi (median per kelas){RST}")
print(f"  {G}✅ Normalisasi (StandardScaler) selesai{RST}")
line(); pause()

# ── STEP 4 ──────────────────────────────────────────────────────
header(4, "Split Data (80% latih / 20% uji)", "✂️")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"  {G}✅ Data latih  : {len(X_train)} baris{RST}")
print(f"  {G}✅ Data uji    : {len(X_test)} baris{RST}")
line(); pause()

# ── STEP 5 ──────────────────────────────────────────────────────
header(5, "Training Model Random Forest", "🌲")
print(f"  {Y}⏳ Melatih model, harap tunggu...{RST}")
model = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=4,
    min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(f"  {G}✅ Training selesai!{RST}")
line(); pause()

# ── STEP 6 ──────────────────────────────────────────────────────
header(6, "Evaluasi Model", "📊")
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\n  {W}Akurasi Model : {G}{acc*100:.2f}%{RST}\n")

report = classification_report(y_test, y_pred,
    target_names=["Tidak Layak", "Layak Minum"], digits=2)
print(f"{C}Laporan Klasifikasi:{RST}")
line()
for baris in report.split("\n"):
    if "Tidak Layak" in baris or "Layak Minum" in baris:
        print(f"  {G}{baris}{RST}")
    elif "accuracy" in baris:
        print(f"  {Y}{baris}{RST}")
    elif "macro" in baris or "weighted" in baris:
        print(f"  {B}{baris}{RST}")
    else:
        print(f"  {W}{baris}{RST}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n{C}Confusion Matrix:{RST}")
line()
labels = ["Tidak Layak", "Layak Minum"]
print(f"  {W}{'':>14}" + "".join(f"{lb:>14}" for lb in labels) + RST)
for i, row in enumerate(cm):
    row_str = f"  {W}{labels[i]:>14}{RST}"
    for j, val in enumerate(row):
        clr = G if i == j else R
        row_str += f"{clr}{val:>14}{RST}"
    print(row_str)
line(); pause()

# ── STEP 7 ──────────────────────────────────────────────────────
header(7, "Menyimpan Model", "💾")
with open(os.path.join(BASE_DIR, "rf_model.pkl"), "wb") as f:    pickle.dump(model, f)
with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:      pickle.dump(scaler, f)
with open(os.path.join(BASE_DIR, "feature_names.pkl"), "wb") as f: pickle.dump(list(X.columns), f)
print(f"  {G}✅ rf_model.pkl tersimpan{RST}")
print(f"  {G}✅ scaler.pkl tersimpan{RST}")
print(f"  {G}✅ feature_names.pkl tersimpan{RST}")
line(); pause()

# ── DETAIL MODEL ─────────────────────────────────────────────────
print(f"\n  {Y}🔶 DETAIL MODEL RANDOM FOREST{RST}")
line()
print(f"  {DIM}•{RST} Jumlah pohon (n_estimators)  : {G}{model.n_estimators}{RST}")
print(f"  {DIM}•{RST} Kedalaman maksimum (max_depth): {G}{model.max_depth}{RST}")
print(f"  {DIM}•{RST} Jumlah fitur yang dipakai     : {G}{model.n_features_in_}{RST}")
print(f"  {DIM}•{RST} Jumlah kelas                  : {G}{model.n_classes_}{RST}")
print(f"  {DIM}•{RST} Kelas yang diprediksi         : {G}{list(model.classes_)}{RST}")

print(f"\n  {G}🌿 Feature Importance (Fitur Paling Berpengaruh):{RST}")
line()
feat_imp = sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
max_imp = feat_imp[0][1]
for i, (feat, imp) in enumerate(feat_imp, 1):
    bar = "█" * int((imp / max_imp) * 20)
    print(f"  {W}{i:>2}. {feat:<25}{RST} {G}{imp:.4f}{RST}  {C}{bar}{RST}")

line()
print(f"\n  {M}🎉 Training selesai! Jalankan: python app.py{RST}\n")