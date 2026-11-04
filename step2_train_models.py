"""
===================================================
TUGAS AKHIR - Step 2: Pelatihan Model 4 Skenario & SHAP (3 Negara)
===================================================
Sesuai Catatan Revisi Bimbingan Dosen:
    1. Skenario 1: Profil Saja (Bintang, Review, Harga) -> Kondisi Sebelum
    2. Skenario 2: Profil + Kalender (Hari Libur, Weekend, Bulan)
    3. Skenario 3: Profil + Destinasi Wisata Sekitar (5 km)
    4. Skenario 4: Profil + Destinasi + Kalender -> Kondisi Sesudah (Lengkap)

Pembagian Data:
    80% Data Latih (Train) : 20% Data Uji (Testing)
===================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from catboost import CatBoostRegressor
import shap

print("="*60)
print("STEP 2A: Memuat Master Dataset 3 Negara...")
print("="*60)

df = pd.read_csv('dataset/master_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Total baris data : {len(df):,}")
print(f"Rentang tanggal  : {df['date'].min().date()} s/d {df['date'].max().date()}")
print(f"Distribusi negara:")
print(df['country'].value_counts().to_string())

# Imputasi nilai kosong jika ada
df['star_rating'] = df['star_rating'].fillna(df['star_rating'].median())
df['guest_rating_score'] = df['guest_rating_score'].fillna(df['guest_rating_score'].median())
df['reference_price'] = df['reference_price'].fillna(df['reference_price'].median())

# ============================================================
# DEFINISI 4 SKENARIO SESUAI CATATAN DOSEN
# ============================================================
features_profile = ['star_rating', 'guest_rating_score', 'reference_price']

features_calendar = [
    'day_of_week', 'is_weekend', 'month',
    'is_holiday', 'is_pre_holiday', 'is_post_holiday'
]

features_destination = [
    'num_destinations_5km', 'avg_rating_5km', 'max_rating_5km',
    'attraction_density_5km', 'distance_weighted_exposure'
]

scenarios = {
    'Skenario 1 (Profil Hotel Saja)': features_profile,
    'Skenario 2 (Profil + Kalender)': features_profile + features_calendar,
    'Skenario 3 (Profil + Destinasi)': features_profile + features_destination,
    'Skenario 4 (Profil + Destinasi + Kalender)': features_profile + features_destination + features_calendar
}

# ============================================================
# PEMBAGIAN DATA: 80% TRAIN : 20% TEST
# ============================================================
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

y_train = train_df['demand']
y_test = test_df['demand']

print(f"\nData Latih (Train 80%) : {len(train_df):,} baris ({train_df['date'].min().date()} s/d {train_df['date'].max().date()})")
print(f"Data Uji   (Test 20%)  : {len(test_df):,} baris ({test_df['date'].min().date()} s/d {test_df['date'].max().date()})")

# ============================================================
# STEP 2B: PELATIHAN 4 SKENARIO (RANDOM FOREST, XGBOOST, CATBOOST)
# ============================================================
print("\n" + "="*60)
print("STEP 2B: Melatih Model pada 4 Skenario Bimbingan...")
print("="*60)

results = []
trained_models = {}

for sc_name, sc_features in scenarios.items():
    X_train = train_df[sc_features]
    X_test = test_df[sc_features]

    # 1. Random Forest (Baseline)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # 2. XGBoost
    xg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbosity=0)
    xg.fit(X_train, y_train)
    xg_pred = xg.predict(X_test)

    # 3. CatBoost
    cb = CatBoostRegressor(iterations=100, learning_rate=0.05, depth=5, random_seed=42, verbose=0)
    cb.fit(X_train, y_train)
    cb_pred = cb.predict(X_test)

    for m_name, pred, model_obj in [('Random Forest', rf_pred, rf), ('XGBoost', xg_pred, xg), ('CatBoost', cb_pred, cb)]:
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        results.append({
            'Skenario': sc_name,
            'Model': m_name,
            'Jumlah Fitur': len(sc_features),
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2 Score': round(r2, 4)
        })
        trained_models[f"{sc_name}_{m_name}"] = (model_obj, sc_features)

df_results = pd.DataFrame(results)
print("\nHASIL TABEL ABLATION STUDY (4 SKENARIO):")
print(df_results.to_string(index=False))

df_results.to_csv('dataset/ablation_results.csv', index=False)
print("\n[INFO] Tabel hasil evaluasi disimpan ke: dataset/ablation_results.csv")

# Simpan model-model terbaik ke joblib untuk disajikan di web
model_bundle = {
    'features_profile': features_profile,
    'features_calendar': features_calendar,
    'features_destination': features_destination,
    'scenarios': scenarios,
    'm1_model': trained_models['Skenario 1 (Profil Hotel Saja)_XGBoost'][0],
    'm2_model': trained_models['Skenario 2 (Profil + Kalender)_XGBoost'][0],
    'm3_model': trained_models['Skenario 3 (Profil + Destinasi)_XGBoost'][0],
    'm4_model': trained_models['Skenario 4 (Profil + Destinasi + Kalender)_XGBoost'][0]
}
joblib.dump(model_bundle, 'dataset/models_4scenarios.pkl')
print("[INFO] Model 4 skenario berhasil disimpan ke: dataset/models_4scenarios.pkl")

# ============================================================
# STEP 2C: ANALISIS SHAP (MENJAWAB PERTANYAAN DOSEN)
# ============================================================
print("\n" + "="*60)
print("STEP 2C: Menghitung Kontribusi Fitur (SHAP Analysis)...")
print("="*60)

best_model = model_bundle['m4_model']
full_features = scenarios['Skenario 4 (Profil + Destinasi + Kalender)']
X_full_test = test_df[full_features]

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_full_test)

mean_shap = np.abs(shap_values).mean(axis=0)
df_shap = pd.DataFrame({
    'Fitur': full_features,
    'SHAP_Importance': mean_shap
}).sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)

cat_profile = sum(df_shap[df_shap['Fitur'].isin(features_profile)]['SHAP_Importance'])
cat_destination = sum(df_shap[df_shap['Fitur'].isin(features_destination)]['SHAP_Importance'])
cat_calendar = sum(df_shap[df_shap['Fitur'].isin(features_calendar)]['SHAP_Importance'])

total_impact = cat_profile + cat_destination + cat_calendar
pct_profile = (cat_profile / total_impact) * 100
pct_destination = (cat_destination / total_impact) * 100
pct_calendar = (cat_calendar / total_impact) * 100

print("\n" + "="*50)
print("JAWABAN UNTUK PERTANYAAN DOSEN:")
print("Mana yang lebih berpengaruh terhadap Demand Hotel?")
print("="*50)
print(f"1. Profil Dasar Hotel (Harga/Bintang/Review) : {pct_profile:.2f}%")
print(f"2. Destinasi Wisata Sekitar (Spasial)        : {pct_destination:.2f}%")
print(f"3. Kalender Hari Libur / Temporal             : {pct_calendar:.2f}%")
print("="*50)

if pct_destination > pct_calendar:
    print(f"KESIMPULAN: Destinasi wisata sekitar memiliki pengaruh LEBIH BESAR ({pct_destination:.2f}%) dibandingkan kalender hari libur ({pct_calendar:.2f}%)!")
else:
    print(f"KESIMPULAN: Kalender hari libur memiliki pengaruh LEBIH BESAR ({pct_calendar:.2f}%) dibandingkan destinasi wisata sekitar ({pct_destination:.2f}%)!")

# Simpan Grafik Visualisasi SHAP
try:
    plt.figure(figsize=(9, 5.5))
    categories = ['Destinasi Wisata Sekitar', 'Kalender Hari Libur', 'Profil Dasar Hotel']
    values = [pct_destination, pct_calendar, pct_profile]
    colors = ['#2563EB', '#F59E0B', '#6B7280']

    bars = plt.bar(categories, values, color=colors, width=0.45)
    plt.title('Perbandingan Kontribusi Fitur terhadap Demand Hotel (Analisis SHAP)', fontsize=13, fontweight='bold')
    plt.ylabel('Kontribusi Relatif (%)', fontsize=11)
    plt.ylim(0, max(values) + 12)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    chart_path = 'dataset/pengaruh_fitur_shap.png'
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\n[INFO] Grafik perbandingan disimpan ke: {chart_path}")
except Exception as e:
    print("Gagal menyimpan grafik:", e)

print("\n[SELESAI] Step 2 Berhasil Diselesaikan!")
