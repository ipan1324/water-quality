"""
Script training otomatis (tanpa pause) - untuk Railway deployment
"""
import os, pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "water_potability.csv")

print("🌲 Training Random Forest...")

df = pd.read_csv(CSV_PATH)

# Handle missing values
for col in df.columns[df.isnull().any()]:
    for cls in df["Potability"].unique():
        med = df.loc[df["Potability"] == cls, col].median()
        df.loc[(df["Potability"] == cls) & (df[col].isnull()), col] = med

X = df.drop("Potability", axis=1)
y = df["Potability"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

model = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=4,
    min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
model.fit(X_train_sc, y_train)

with open(os.path.join(BASE_DIR, "rf_model.pkl"), "wb") as f:    pickle.dump(model, f)
with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:      pickle.dump(scaler, f)
with open(os.path.join(BASE_DIR, "feature_names.pkl"), "wb") as f: pickle.dump(list(X.columns), f)

print("✅ Model berhasil disimpan!")