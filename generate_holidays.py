"""
Generate Holiday Dataset untuk TA Hotel Demand Forecasting
==========================================================
Output: dataset/holidays.csv
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import holidays
import pandas as pd
from datetime import date, timedelta

# ============================================================
# KONFIGURASI
# ============================================================
YEARS = [2024, 2025, 2026]  # Sesuai range dataset ICOICT

# Daftar negara yang di-cover
# Format: (country_code, country_name_in_dataset)
COUNTRIES = [
    ("AU", "Australia"),
    ("US", "United States Of America"),
    ("GB", "United Kingdom Of Great Britain And Northern Ireland"),
    ("IT", "Italy"),
    ("FR", "France"),
    ("ES", "Spain"),
    ("DE", "Germany"),
    ("CN", "China"),
    ("JP", "Japan"),
    ("TH", "Thailand"),
    ("MX", "Mexico"),
    ("ID", "Indonesia"),
]

# ============================================================
# GENERATE HOLIDAYS
# ============================================================
print("Generating holiday data...")
all_holidays = []

for country_code, country_name in COUNTRIES:
    try:
        h = holidays.country_holidays(country_code, years=YEARS)
        for holiday_date, holiday_name in sorted(h.items()):
            all_holidays.append({
                "date": holiday_date,
                "holiday_name": holiday_name,
                "country_code": country_code,
                "country_name": country_name,
                "is_holiday": 1
            })
        print(f"  [OK] {country_name}: {len(h)} hari libur")
    except Exception as e:
        print(f"  [ERROR] {country_name}: Error - {e}")

# ============================================================
# BUAT DATAFRAME & SIMPAN
# ============================================================
df_holidays = pd.DataFrame(all_holidays)
df_holidays["date"] = pd.to_datetime(df_holidays["date"])
df_holidays = df_holidays.sort_values(["country_code", "date"]).reset_index(drop=True)

# Simpan ke folder dataset
output_path = "dataset/holidays.csv"
df_holidays.to_csv(output_path, index=False)

# ============================================================
# LAPORAN
# ============================================================
print("\n" + "="*50)
print("[SELESAI] File tersimpan di:", output_path)
print("="*50)
print(f"Total baris  : {len(df_holidays):,}")
print(f"Total negara : {df_holidays['country_code'].nunique()}")
print(f"Range waktu  : {df_holidays['date'].min().date()} s/d {df_holidays['date'].max().date()}")
print()
print("Preview 10 baris pertama:")
print(df_holidays.head(10).to_string(index=False))
print()
print("Jumlah hari libur per negara:")
print(df_holidays.groupby("country_name")["holiday_name"].count().sort_values(ascending=False).to_string())
