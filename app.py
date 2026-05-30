"""
Flask Web Application - Klasifikasi Kualitas Air Minum
Menggunakan Algoritma Random Forest
Author: Tugas 8 - Kecerdasan Buatan
"""

import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load model artefacts ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'model', 'rf_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'model', 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, 'model', 'feature_names.pkl'), 'rb') as f:
    feature_names = pickle.load(f)

# ── Feature metadata untuk UI ──────────────────────────────────────────────────
FEATURES = [
    {
        "key": "ph",
        "label": "pH Air",
        "unit": "",
        "min": 0, "max": 14, "step": 0.1,
        "default": 7.0,
        "safe_range": "6.5 – 8.5",
        "description": "Tingkat keasaman / kebasaan air",
        "icon": "⚗️"
    },
    {
        "key": "Hardness",
        "label": "Kekerasan (Hardness)",
        "unit": "mg/L",
        "min": 50, "max": 400, "step": 1,
        "default": 180,
        "safe_range": "< 200 mg/L",
        "description": "Kandungan kalsium dan magnesium",
        "icon": "🪨"
    },
    {
        "key": "Solids",
        "label": "Total Padatan Terlarut",
        "unit": "ppm",
        "min": 1000, "max": 50000, "step": 100,
        "default": 15000,
        "safe_range": "< 20000 ppm",
        "description": "Total dissolved solids dalam air",
        "icon": "🧪"
    },
    {
        "key": "Chloramines",
        "label": "Kloramin",
        "unit": "ppm",
        "min": 0, "max": 15, "step": 0.1,
        "default": 6.5,
        "safe_range": "< 8 ppm",
        "description": "Desinfektan berbasis klor-amonia",
        "icon": "🧫"
    },
    {
        "key": "Sulfate",
        "label": "Sulfat",
        "unit": "mg/L",
        "min": 100, "max": 600, "step": 1,
        "default": 300,
        "safe_range": "< 350 mg/L",
        "description": "Konsentrasi ion sulfat dalam air",
        "icon": "⚡"
    },
    {
        "key": "Conductivity",
        "label": "Konduktivitas",
        "unit": "μS/cm",
        "min": 100, "max": 900, "step": 1,
        "default": 400,
        "safe_range": "< 500 μS/cm",
        "description": "Kemampuan air menghantarkan listrik",
        "icon": "🔋"
    },
    {
        "key": "Organic_carbon",
        "label": "Karbon Organik",
        "unit": "ppm",
        "min": 2, "max": 30, "step": 0.1,
        "default": 12,
        "safe_range": "< 15 ppm",
        "description": "Total organic carbon (TOC) dalam air",
        "icon": "🌿"
    },
    {
        "key": "Trihalomethanes",
        "label": "Trihalometan",
        "unit": "μg/L",
        "min": 10, "max": 130, "step": 0.5,
        "default": 60,
        "safe_range": "< 80 μg/L",
        "description": "Produk sampingan klorinasi air",
        "icon": "☣️"
    },
    {
        "key": "Turbidity",
        "label": "Turbiditas (Kekeruhan)",
        "unit": "NTU",
        "min": 1, "max": 10, "step": 0.1,
        "default": 3.5,
        "safe_range": "< 4 NTU",
        "description": "Tingkat kekeruhan / kejernihan air",
        "icon": "💧"
    },
]

# ── Feature importance dari model ───────────────────────────────────────────────
importances = model.feature_importances_
feat_imp_data = [
    {"feature": fn, "importance": round(float(imp), 4)}
    for fn, imp in sorted(zip(feature_names, importances),
                          key=lambda x: x[1], reverse=True)
]

# ── Routes ──────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           features=FEATURES,
                           feat_imp=feat_imp_data)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        values = [float(data[f['key']]) for f in FEATURES]
        arr    = np.array(values).reshape(1, -1)
        arr_df = pd.DataFrame([values], columns=feature_names)
        arr_sc = scaler.transform(arr_df)

        prediction   = int(model.predict(arr_sc)[0])
        proba        = model.predict_proba(arr_sc)[0]
        confidence   = round(float(max(proba)) * 100, 2)
        proba_layak  = round(float(proba[1]) * 100, 2)
        proba_tidak  = round(float(proba[0]) * 100, 2)

        # Analisis per parameter
        param_analysis = []
        for feat, val in zip(FEATURES, values):
            safe = _is_safe(feat['key'], val)
            param_analysis.append({
                "label": feat['label'],
                "value": val,
                "unit": feat['unit'],
                "safe_range": feat['safe_range'],
                "safe": safe,
                "icon": feat['icon']
            })

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "label": "Layak Minum ✅" if prediction == 1 else "Tidak Layak ❌",
            "confidence": confidence,
            "proba_layak": proba_layak,
            "proba_tidak": proba_tidak,
            "param_analysis": param_analysis
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


def _is_safe(key, val):
    safe_map = {
        "ph":              (6.5, 8.5),
        "Hardness":        (0,   200),
        "Solids":          (0, 20000),
        "Chloramines":     (0,     8),
        "Sulfate":         (0,   350),
        "Conductivity":    (0,   500),
        "Organic_carbon":  (0,    15),
        "Trihalomethanes": (0,    80),
        "Turbidity":       (0,     4),
    }
    lo, hi = safe_map.get(key, (None, None))
    if lo is None:
        return True
    return lo <= val <= hi


if __name__ == '__main__':
    app.run(debug=True, port=5000)
