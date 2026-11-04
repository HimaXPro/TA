"""
===================================================
TUGAS AKHIR - Step 1: Preprocessing 3 Negara (AU, UK, ID)
===================================================
Tujuan:
    Menggabungkan data transaksi hotel, profil hotel, hari libur nasional,
    dan destinasi wisata sekitar (radius 5km) untuk 3 negara:
    1. Australia (AU)
    2. United Kingdom / Inggris (GB)
    3. Indonesia (ID)

Output:
    dataset/master_dataset.csv
===================================================
"""

import pandas as pd
import numpy as np

print("="*60)
print("STEP 1A: Memuat Data Transaksi Hotel...")
print("="*60)

df_trans = pd.read_excel('dataset/Accommodation - Transactions by HotelCode.xlsx')
df_trans.rename(columns={
    'payment_date': 'date',
    'Property Id': 'hotel_id',
    'Product': 'product_type',
    'Number Transactions': 'demand'
}, inplace=True)
df_trans['date'] = pd.to_datetime(df_trans['date'])

print(f"Total baris transaksi : {len(df_trans):,}")
print(f"Rentang tanggal       : {df_trans['date'].min().date()} s/d {df_trans['date'].max().date()}")
print(f"Hotel unik            : {df_trans['hotel_id'].nunique():,}")

# ============================================================
# STEP 1B: Memuat Profil Hotel (accommodations.csv) untuk 3 Negara
# ============================================================
print("\n" + "="*60)
print("STEP 1B: Menyaring Hotel untuk 3 Negara (AU, UK, ID)...")
print("="*60)

hotel_ids_needed = set(df_trans['hotel_id'].unique())
COLS_ACC = ['id', 'name', 'country', 'city', 'star_rating', 'guest_rating_score',
            'reference_price', 'destination_coordinate']

def standardize_country(c):
    c_str = str(c).strip()
    if 'australia' in c_str.lower():
        return 'Australia'
    elif any(k in c_str.lower() for k in ['united kingdom', 'great britain', 'england', 'scotland', 'wales']):
        return 'United Kingdom'
    elif 'indonesia' in c_str.lower():
        return 'Indonesia'
    return None

def parse_coord(coord_str):
    try:
        clean = str(coord_str).replace("'", "").replace('"', "").strip()
        parts = clean.split(',')
        return float(parts[0].strip()), float(parts[1].strip())
    except:
        return np.nan, np.nan

acc_chunks = []
chunk_count = 0
for chunk in pd.read_csv('dataset/accommodations.csv', usecols=COLS_ACC, chunksize=100000):
    matched = chunk[chunk['id'].isin(hotel_ids_needed)].copy()
    if not matched.empty:
        matched['country_std'] = matched['country'].apply(standardize_country)
        matched_3c = matched[matched['country_std'].notna()]
        if not matched_3c.empty:
            acc_chunks.append(matched_3c)
    chunk_count += 1
    if chunk_count % 10 == 0:
        print(f"  Memproses chunk ke-{chunk_count}... (hotel terkumpul: {sum(len(c) for c in acc_chunks)})")

df_acc = pd.concat(acc_chunks, ignore_index=True)
df_acc.rename(columns={'id': 'hotel_id', 'country_std': 'country_clean'}, inplace=True)
df_acc.drop_duplicates(subset='hotel_id', inplace=True)

# Parse koordinat hotel
coords = [parse_coord(c) for c in df_acc['destination_coordinate']]
df_acc['hotel_lat'] = [c[0] for c in coords]
df_acc['hotel_lon'] = [c[1] for c in coords]
df_acc.drop(columns=['destination_coordinate', 'country'], inplace=True)
df_acc.rename(columns={'country_clean': 'country'}, inplace=True)

# Bersihkan dan standarisasi nama kota
df_acc['city'] = df_acc['city'].astype(str).str.strip().str.title()

print(f"Total profil hotel 3 negara : {len(df_acc):,}")
print("Distribusi per negara:")
print(df_acc['country'].value_counts().to_string())

# ============================================================
# STEP 1C: Gabungkan Transaksi + Profil Hotel
# ============================================================
print("\n" + "="*60)
print("STEP 1C: Menggabungkan Transaksi + Profil Hotel...")
print("="*60)

df_main = df_trans.merge(df_acc, on='hotel_id', how='inner')
print(f"Baris setelah merge : {len(df_main):,}")
print(f"Hotel unik          : {df_main['hotel_id'].nunique():,}")

# ============================================================
# STEP 1D: Feature Engineering - Fitur Temporal & Hari Libur 3 Negara
# ============================================================
print("\n" + "="*60)
print("STEP 1D: Mengintegrasikan Hari Libur Nasional (AU, UK, ID)...")
print("="*60)

df_main['day_of_week']   = df_main['date'].dt.dayofweek
df_main['is_weekend']    = df_main['day_of_week'].isin([5, 6]).astype(int)
df_main['month']         = df_main['date'].dt.month
df_main['week_of_year']  = df_main['date'].dt.isocalendar().week.astype(int)
df_main['quarter']       = df_main['date'].dt.quarter
df_main['day_of_year']   = df_main['date'].dt.dayofyear

# Mapping kode negara untuk holidays.csv: Australia -> AU, United Kingdom -> GB, Indonesia -> ID
country_code_map = {
    'Australia': 'AU',
    'United Kingdom': 'GB',
    'Indonesia': 'ID'
}
df_main['country_code'] = df_main['country'].map(country_code_map)

df_hol = pd.read_csv('dataset/holidays.csv')
df_hol['date'] = pd.to_datetime(df_hol['date'])
df_hol_3c = df_hol[df_hol['country_code'].isin(['AU', 'GB', 'ID'])][['date', 'country_code', 'holiday_name', 'is_holiday']]
df_hol_3c.rename(columns={'holiday_name': 'holiday_event_name'}, inplace=True)

df_main = df_main.merge(df_hol_3c, on=['date', 'country_code'], how='left')
df_main['is_holiday'] = df_main['is_holiday'].fillna(0).astype(int)
df_main['holiday_event_name'] = df_main['holiday_event_name'].fillna('None')

# Window H-1 dan H+1 per negara
holiday_lookup = set(zip(df_hol_3c['country_code'], df_hol_3c['date']))

df_main['is_pre_holiday'] = df_main.apply(
    lambda r: 1 if (r['country_code'], r['date'] + pd.Timedelta(days=1)) in holiday_lookup else 0, axis=1
)
df_main['is_post_holiday'] = df_main.apply(
    lambda r: 1 if (r['country_code'], r['date'] - pd.Timedelta(days=1)) in holiday_lookup else 0, axis=1
)

print(f"Total baris yang jatuh pada hari libur nasional: {df_main['is_holiday'].sum()}")

# ============================================================
# STEP 1E: Feature Engineering - Destinasi Wisata Sekitar (Radius 5 km)
# ============================================================
print("\n" + "="*60)
print("STEP 1E: Menghitung Destinasi Wisata Sekitar (Radius 5 km)...")
print("="*60)

def haversine_vectorized(lat1, lon1, lats_arr, lons_arr):
    R = 6371.0
    dlat = np.radians(lats_arr - lat1)
    dlon = np.radians(lons_arr - lon1)
    a = (np.sin(dlat / 2.0)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lats_arr)) * np.sin(dlon / 2.0)**2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

COLS_ACT = ['name', 'rating', 'destination_coordinate', 'destination_country']
RADIUS_KM = 5.0

act_chunks = []
for chunk in pd.read_csv('dataset/activity.csv', usecols=COLS_ACT, chunksize=100000,
                          encoding='utf-8', encoding_errors='replace', on_bad_lines='skip'):
    chunk['country_std'] = chunk['destination_country'].apply(standardize_country)
    m = chunk[chunk['country_std'].notna()].copy()
    if not m.empty:
        act_chunks.append(m)

df_act = pd.concat(act_chunks, ignore_index=True)

# Parse koordinat wisata
act_coords = [parse_coord(c) for c in df_act['destination_coordinate']]
df_act['act_lat'] = [c[0] for c in act_coords]
df_act['act_lon'] = [c[1] for c in act_coords]
df_act['rating'] = pd.to_numeric(df_act['rating'], errors='coerce').fillna(0.0)
df_act = df_act.dropna(subset=['act_lat', 'act_lon'])

print(f"Total destinasi wisata valid (3 negara): {len(df_act):,}")
for c in ['Australia', 'United Kingdom', 'Indonesia']:
    cnt = len(df_act[df_act['country_std'] == c])
    print(f"  - {c}: {cnt:,} lokasi")

# Hitung jarak per hotel unik sesuai negaranya
hotel_coords = df_main[['hotel_id', 'country', 'hotel_lat', 'hotel_lon']].drop_duplicates()
spatial_features = []

print(f"\nMenghitung fitur spasial untuk {len(hotel_coords)} hotel...")
for idx, (_, row) in enumerate(hotel_coords.iterrows()):
    h_id = row['hotel_id']
    h_country = row['country']
    h_lat = row['hotel_lat']
    h_lon = row['hotel_lon']

    if pd.isna(h_lat) or pd.isna(h_lon):
        spatial_features.append({
            'hotel_id': h_id,
            'num_destinations_5km': 0,
            'avg_rating_5km': 0.0,
            'max_rating_5km': 0.0,
            'attraction_density_5km': 0.0,
            'distance_weighted_exposure': 0.0
        })
        continue

    # Filter destinasi hanya di negara yang sama untuk efisiensi
    country_acts = df_act[df_act['country_std'] == h_country]
    if country_acts.empty:
        spatial_features.append({
            'hotel_id': h_id,
            'num_destinations_5km': 0,
            'avg_rating_5km': 0.0,
            'max_rating_5km': 0.0,
            'attraction_density_5km': 0.0,
            'distance_weighted_exposure': 0.0
        })
        continue

    act_lats = country_acts['act_lat'].values
    act_lons = country_acts['act_lon'].values
    act_ratings = country_acts['rating'].values

    dists = haversine_vectorized(h_lat, h_lon, act_lats, act_lons)
    nearby_mask = dists <= RADIUS_KM
    n_nearby = np.sum(nearby_mask)

    if n_nearby > 0:
        nearby_ratings = act_ratings[nearby_mask]
        nearby_dists = dists[nearby_mask]
        spatial_features.append({
            'hotel_id': h_id,
            'num_destinations_5km': int(n_nearby),
            'avg_rating_5km': float(np.mean(nearby_ratings)),
            'max_rating_5km': float(np.max(nearby_ratings)),
            'attraction_density_5km': float(n_nearby / (np.pi * (RADIUS_KM**2))),
            'distance_weighted_exposure': float(np.sum(nearby_ratings / (nearby_dists + 0.1)))
        })
    else:
        spatial_features.append({
            'hotel_id': h_id,
            'num_destinations_5km': 0,
            'avg_rating_5km': 0.0,
            'max_rating_5km': 0.0,
            'attraction_density_5km': 0.0,
            'distance_weighted_exposure': 0.0
        })

df_spatial = pd.DataFrame(spatial_features)
df_main = df_main.merge(df_spatial, on='hotel_id', how='left')

# ============================================================
# STEP 1F: Simpan Master Dataset 3 Negara
# ============================================================
print("\n" + "="*60)
print("STEP 1F: Menyimpan Master Dataset...")
print("="*60)

df_main.drop(columns=['product_type', 'country_code'], inplace=True, errors='ignore')
output_path = 'dataset/master_dataset.csv'
df_main.to_csv(output_path, index=False)

print(f"[SELESAI] Master dataset 3 negara berhasil disimpan ke: {output_path}")
print(f"Total baris : {len(df_main):,}")
print(f"Total kolom : {len(df_main.columns)}")
print("Distribusi baris per negara:")
print(df_main['country'].value_counts().to_string())
